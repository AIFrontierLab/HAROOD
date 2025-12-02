import torch
import numpy as np
import argparse
from scipy.stats import wasserstein_distance
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
from datautil.getdataloader import get_act_dataloader, get_aug_dataloaders
import datautil.actdata.cross_dataset as cross_dataset
import datautil.actdata.cross_people as cross_people
import datautil.actdata.cross_position as cross_position
import datautil.actdata.cross_time as cross_time
import datautil.actdata.util as actutil
import matplotlib.pyplot as plt

task_act = {
    "cross_dataset": cross_dataset,
    "cross_people": cross_people,
    "cross_position": cross_position,
    "cross_time": cross_time,
}


def get_act_datax(args):
    pcross_act = task_act[args.task]
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


def get_args():
    parser = argparse.ArgumentParser(description="DG")
    parser.add_argument("--data_file", type=str, default="", help="root_dir")
    parser.add_argument("--dataset", type=str, default="office")
    parser.add_argument("--data_dir", type=str, default="", help="data dir")
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="0", help="device id to run"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--task",
        type=str,
        default="cross_people",
        choices=["cross_dataset", "cross_people", "cross_position", "cross_time"],
        help="now only support image tasks",
    )
    args = parser.parse_args()
    args.steps_per_epoch = 100
    args.data_dir = args.data_file + args.data_dir
    args.test_envs = [100]
    args = act_param_init(args)
    return args


args = get_args()
domains = get_act_datax(args)

for dom in range(4):
    tsample = 20
    labels = domains[dom].labels
    indexs = np.squeeze(np.where(labels == 0)[0])
    print(indexs.shape)
    print(indexs[tsample])
    signal = domains[dom].x[indexs[tsample]][0][0][:]  # Shape: (len,)

    if isinstance(signal, torch.Tensor):
        signal = signal.numpy()
    x = np.arange(len(signal))  # Create an x-axis for the signal
    print(signal.shape)
    print(signal)
    # Plot the 1D signal
    plt.figure(figsize=(12, 4))
    plt.plot(x, signal)
    plt.grid(True, alpha=0.3)
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(
        "./analys/datavis/signal-%s-%s-%d.jpg" % (args.task, args.dataset, dom),
        bbox_inches="tight",
        dpi=300,
        format="jpg",
        transparent=False,
    )
    plt.close()
