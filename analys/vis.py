import argparse
import os
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from alg.opt import *
from alg import alg
from utils.util import (
    set_random_seed,
    save_checkpoint,
    print_args,
    train_valid_target_eval_names,
    alg_loss_dict,
    Tee,
    print_environ,
    act_param_init,
    get_str_from_args,
)
from datautil.getdataloader import (
    get_img_dataloader,
    get_act_dataloader,
    get_aug_dataloaders,
)
import datautil.actdata.cross_dataset as cross_dataset
import datautil.actdata.cross_people as cross_people
import datautil.actdata.cross_position as cross_position
import datautil.actdata.cross_time as cross_time
import datautil.actdata.util as actutil

# python -m analys.vis --task cross_people --dataset dsads --test_envs 0 --output_dir /data_2/lw/DeepDG/output/ --data_dir ./data/

task_act = {
    "cross_dataset": cross_dataset,
    "cross_people": cross_people,
    "cross_position": cross_position,
    "cross_time": cross_time,
}


def get_act_datax(args):
    pcross_act = task_act[args.task]
    rate = 0.2
    if args.task == "cross_people" or args.task == "cross_time":
        tmpp = args.act_people[args.dataset]
    elif args.task == "cross_position":
        tmpp = args.act_positon[args.dataset]
    else:
        tmpp = args.act_dataset
    args.domain_num = len(tmpp)
    tedatalist = []
    for i, item in enumerate(tmpp):
        tedatalist.append(
            pcross_act.ActList(
                args, args.dataset, args.data_dir, item, i, transform=actutil.act_test()
            )
        )
    return tedatalist


def get_best_valid_folder(base_dir):
    best_acc = -1
    best_folder = None

    for folder in os.listdir(base_dir):
        done_file = os.path.join(base_dir, folder, "done.txt")
        if os.path.exists(done_file):
            with open(done_file, "r") as f:
                lines = f.readlines()
                # Extract validation accuracy from the file
                for line in lines:
                    if "valid acc:" in line:
                        try:
                            valid_acc = float(line.split("valid acc:")[1].strip())
                            if valid_acc > best_acc:
                                best_acc = valid_acc
                                best_folder = os.path.join(base_dir, folder)
                        except ValueError:
                            continue
    return best_folder


def featurize_batch(args, model, batch_x):
    # Assuming model has a method `featurizer` to extract features
    with torch.no_grad():
        if args.algorithm == "LAG":
            features = model.extract_features(batch_x.cuda().float())
        elif args.algorithm == "Fish":
            features = model.network.featurizer(batch_x.cuda().float())
        else:
            features = model.featurizer(batch_x.cuda().float())
    return features.cpu().numpy()


def perform_tsne(features):
    tsne = TSNE(n_components=2, random_state=42)
    transformed_data = tsne.fit_transform(features)
    return transformed_data


def visualize_tsne(ax, transformed_data, labels, dlabels, title=""):
    markers = ["o", "s", "^", "D", "*"]
    colors = np.array(
        [
            "red",
            "blue",
            "green",
            "purple",
            "orange",
            "brown",
            "pink",
            "gray",
            "cyan",
            "magenta",
            "yellow",
            "lime",
            "teal",
            "olive",
            "navy",
            "gold",
            "salmon",
            "turquoise",
            "crimson",
            "lavender",
        ]
    )

    unique_dlabels = np.unique(dlabels)
    for dlabel in unique_dlabels:
        idx = dlabels == dlabel
        ax.scatter(
            transformed_data[idx, 0],
            transformed_data[idx, 1],
            c=colors[labels[idx]],
            alpha=0.7,
            marker=markers[dlabel % len(markers)],
            label=f"Domain {dlabel}",
        )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-SNE Visualization")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="", help="data dir")
    parser.add_argument("--split_style", type=str, default="strat")
    parser.add_argument(
        "--test_envs", type=int, nargs="+", default=[0], help="target domains"
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--N_WORKERS", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--model_size",
        type=str,
        default="medium",
        choices=["small", "medium", "large", "transformer"],
    )
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument(
        "--classifier", type=str, default="linear", choices=["linear", "wn"]
    )
    parser.add_argument(
        "--dis_hidden", type=int, default=256, help="dis hidden dimension"
    )
    parser.add_argument("--layer", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument(
        "--rsc_f_drop_factor", type=float, default=1 / 3, help="rsc hyper-param"
    )
    parser.add_argument(
        "--rsc_b_drop_factor", type=float, default=1 / 3, help="rsc hyper-param"
    )
    parser.add_argument("--tau", type=float, default=1, help="andmask tau")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--ema", type=float, default=0.95, help="ema hyper-param")
    parser.add_argument("--urm_discriminator_hidden_layers", type=int, default=2)
    parser.add_argument(
        "--urm_generator_output",
        type=str,
        default="tanh",
        choices=["tanh", "relu", "sigmoid", "identity"],
    )
    parser.add_argument("--urm_adv_lambda", type=float, default=0.1)
    parser.add_argument("--urm_discriminator_label_smoothing", type=float, default=0)
    parser.add_argument("--lars", action="store_true")
    args = parser.parse_args()
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()
    algorithmlist = [
        "ERM",
        "Mixup",
        "DDLearn",
        "DANN",
        "CORAL",
        "MMD",
        "VREx",
        "LAG",
        "MLDG",
        "RSC",
        "GroupDRO",
        "ANDMask",
        "Fish",
        "Fishr",
        "URM",
        "ERMPlusPlus",
    ]

    for k, talg in enumerate(algorithmlist):
        args.algorithm = talg
        args.output = os.path.join(
            args.output_dir,
            args.task,
            args.dataset,
            str(args.test_envs[0]),
            args.algorithm,
            str(42),
        )
        args = act_param_init(args)
        # 2. Find valid directories
        train_loaders, eval_loaders = get_act_dataloader(args)
        tmp = args.test_envs[0]
        args.test_envs = [100]
        alldatasets = get_act_datax(args)
        args.test_envs = [tmp]
        algorithm_class = alg.get_algorithm_class(args.algorithm)
        if args.algorithm == "LAG":
            initx = next(zip(train_loaders[0]))
            algorithm = algorithm_class(args, initx[0][0]).cuda()
        else:
            algorithm = algorithm_class(args).cuda()
        if talg == "GroupDRO":
            algorithm.q = torch.ones(args.domain_num - len(args.test_envs)).cuda()
        elif talg == "Fish":
            algorithm.create_clone("cuda")
        algorithm.eval()
        best_folder = get_best_valid_folder(args.output)
        print(best_folder)
        if best_folder:
            model_path = os.path.join(best_folder, "model.pkl")
            state_dict = torch.load(model_path)["model_dict"]
            model_state_dict = algorithm.state_dict()

            filtered_state_dict = {
                k: v
                for k, v in state_dict.items()
                if k in model_state_dict and v.shape == model_state_dict[k].shape
            }

            algorithm.load_state_dict(filtered_state_dict, strict=False)
            all_features = []
            all_labels = []
            all_dlabels = []

            batch_size = 64  # Adjust based on your memory capacity
            for tdata in alldatasets:
                x_batch, labels_batch, dlabels_batch = (
                    tdata.x,
                    tdata.labels,
                    tdata.dlabels,
                )
                num_batches = len(x_batch) // batch_size + (
                    1 if len(x_batch) % batch_size != 0 else 0
                )

                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(x_batch))

                    batch_x = x_batch[start_idx:end_idx]
                    batch_labels = labels_batch[start_idx:end_idx]
                    batch_dlabels = dlabels_batch[start_idx:end_idx]

                    features = featurize_batch(args, algorithm, batch_x)
                    all_features.append(features)
                    all_labels.extend(batch_labels)
                    all_dlabels.extend(batch_dlabels)

            all_features = np.vstack(all_features)
            all_labels = np.array(all_labels, dtype=int)
            all_dlabels = np.array(all_dlabels, dtype=int)
            transformed_data = perform_tsne(all_features)
            visualize_tsne(
                axes[k], transformed_data, all_labels, all_dlabels, title=talg
            )
        print("===%s===" % talg)
    plt.tight_layout()
    plt.savefig(
        "./analys/newvis/tsne-comparison-4x4-%s-%s-%d.pdf"
        % (args.task, args.dataset, args.test_envs[0]),
        bbox_inches="tight",
    )
    plt.savefig(
        "./analys/newvis/c-cpe-d-0.jpg", bbox_inches="tight", dpi=300, format="jpg"
    )
    plt.close()
