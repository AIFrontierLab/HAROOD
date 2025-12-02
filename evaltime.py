# coding=utf-8

import os
import sys
import time
import numpy as np
import argparse

from alg.opt import *
from alg import alg, modelopera
from utils.util import (
    set_random_seed,
    save_checkpoint,
    print_args,
    train_valid_target_eval_names,
    alg_loss_dict,
    Tee,
    img_param_init,
    print_environ,
    act_param_init,
    get_str_from_args,
)
from datautil.getdataloader import (
    get_img_dataloader,
    get_act_dataloader,
    get_aug_dataloaders,
)


def get_args():
    parser = argparse.ArgumentParser(description="DG")
    parser.add_argument("--algorithm", type=str, default="ERM")
    parser.add_argument("--alpha", type=float, default=1, help="DANN dis alpha")
    parser.add_argument(
        "--anneal_iters",
        type=int,
        default=500,
        help="Penalty anneal iters used in VREx",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--beta", type=float, default=1, help="DIFEX beta")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam hyper-param")
    parser.add_argument("--linear_steps", type=int, default=500)
    parser.add_argument("--lars", action="store_true")
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument(
        "--checkpoint_freq", type=int, default=3, help="Checkpoint every N epoch"
    )
    parser.add_argument(
        "--classifier", type=str, default="linear", choices=["linear", "wn"]
    )
    parser.add_argument("--data_file", type=str, default="", help="root_dir")
    parser.add_argument("--dataset", type=str, default="office")
    parser.add_argument("--data_dir", type=str, default="", help="data dir")
    parser.add_argument(
        "--dis_hidden", type=int, default=256, help="dis hidden dimension"
    )
    parser.add_argument(
        "--disttype",
        type=str,
        default="2-norm",
        choices=["1-norm", "2-norm", "cos", "norm-2-norm", "norm-1-norm"],
    )
    parser.add_argument("--distyle", type=str, default="l1", choices=["l1", "l2"])
    parser.add_argument("--urm_discriminator_hidden_layers", type=int, default=2)
    parser.add_argument(
        "--urm_generator_output",
        type=str,
        default="tanh",
        choices=["tanh", "relu", "sigmoid", "identity"],
    )
    parser.add_argument("--urm_adv_lambda", type=float, default=0.1)
    parser.add_argument("--urm_discriminator_label_smoothing", type=float, default=0)
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="0", help="device id to run"
    )
    parser.add_argument("--groupdro_eta", type=float, default=1, help="groupdro eta")
    parser.add_argument(
        "--inner_lr", type=float, default=1e-2, help="learning rate used in MLDG"
    )
    parser.add_argument(
        "--lam", type=float, default=1, help="tradeoff hyperparameter used in VREx"
    )
    parser.add_argument("--layer", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--meta_lr", type=float, default=1e-1)
    parser.add_argument("--lr_decay", type=float, default=0.75, help="for sgd")
    parser.add_argument(
        "--lr_decay1", type=float, default=1.0, help="for pretrained featurizer"
    )
    parser.add_argument(
        "--lr_decay2",
        type=float,
        default=1.0,
        help="inital learning rate decay of network",
    )
    parser.add_argument("--lr_gamma", type=float, default=0.0003, help="for optimizer")
    parser.add_argument("--max_epoch", type=int, default=120, help="max iterations")
    parser.add_argument(
        "--penalty_anneal_iters",
        type=int,
        default=1500,
        help="Penalty anneal iters used in Fishr",
    )
    parser.add_argument("--ema", type=float, default=0.95, help="ema hyper-param")
    parser.add_argument(
        "--mixupalpha", type=float, default=0.2, help="mixup hyper-param"
    )
    parser.add_argument("--mldg_beta", type=float, default=1, help="mldg hyper-param")
    parser.add_argument(
        "--mmd_gamma", type=float, default=1, help="MMD, CORAL hyper-param"
    )
    parser.add_argument(
        "--rela_gamma", type=float, default=1, help="rela gamma, LAG hyper-param"
    )
    parser.add_argument(
        "--auglossweight",
        type=float,
        default=1,
        help="auglossweight, DDLearn hyper-param",
    )
    parser.add_argument(
        "--dpweight", type=float, default=1, help="dpweight, DDLearn hyper-param"
    )
    parser.add_argument(
        "--conweight", type=float, default=1, help="conweight, DDLearn hyper-param"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="for optimizer")
    parser.add_argument(
        "--net",
        type=str,
        default="resnet50",
        help="featurizer: vgg16, resnet50, resnet101,DTNBase",
    )
    parser.add_argument("--N_WORKERS", type=int, default=1)
    parser.add_argument(
        "--rsc_f_drop_factor", type=float, default=1 / 3, help="rsc hyper-param"
    )
    parser.add_argument(
        "--rsc_b_drop_factor", type=float, default=1 / 3, help="rsc hyper-param"
    )
    parser.add_argument("--save_model_every_checkpoint", action="store_true")
    parser.add_argument("--schuse", action="store_true")
    parser.add_argument("--schusech", type=str, default="cos")
    parser.add_argument(
        "--model_size",
        type=str,
        default="medium",
        choices=["small", "medium", "large", "transformer", "rnn", "lstm"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split_style",
        type=str,
        default="strat",
        help="the style to split the train and eval datasets",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="img_dg",
        choices=[
            "img_dg",
            "cross_dataset",
            "cross_people",
            "cross_position",
            "cross_time",
            "cross_device",
        ],
        help="now only support image tasks",
    )
    parser.add_argument("--tau", type=float, default=1, help="andmask tau")
    parser.add_argument(
        "--test_envs", type=int, nargs="+", default=[0], help="target domains"
    )
    parser.add_argument(
        "--output", type=str, default="train_output", help="result output path"
    )
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    args = parser.parse_args()
    args.steps_per_epoch = 100
    args.data_dir = args.data_file + args.data_dir
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if args.model_size == "medium":
        args.output = os.path.join(
            args.output,
            args.task,
            args.dataset,
            str(args.test_envs[0]),
            args.algorithm,
            str(args.seed),
            get_str_from_args(args),
        )
    else:
        args.output = os.path.join(
            args.output,
            args.model_size,
            args.task,
            args.dataset,
            str(args.test_envs[0]),
            args.algorithm,
            str(args.seed),
            get_str_from_args(args),
        )
    if args.task.startswith("cross"):
        args = act_param_init(args)
    else:
        args = img_param_init(args)
    print_environ()
    return args


if __name__ == "__main__":
    args = get_args()
    set_random_seed(args.seed)

    loss_list = alg_loss_dict(args)
    if args.task.startswith("cross"):
        train_loaders, eval_loaders = get_act_dataloader(args)
    else:
        train_loaders, eval_loaders = get_img_dataloader(args)
    algs = [
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
    times = ""
    for algg in algs:
        args.algorithm = algg
        if args.algorithm == "DDLearn":
            aug_train_loaders = get_aug_dataloaders(args, train_loaders)
            aug_train_minibatches_iterator = zip(*aug_train_loaders)
        eval_name_dict = train_valid_target_eval_names(args)
        algorithm_class = alg.get_algorithm_class(args.algorithm)
        if args.algorithm == "LAG":
            initx = next(zip(train_loaders[0]))
            algorithm = algorithm_class(args, initx[0][0]).cuda()
        else:
            algorithm = algorithm_class(args).cuda()
        algorithm.train()
        if not args.algorithm in ["Fishr", "Fish"]:
            opt = get_optimizer(algorithm, args)
        else:
            opt = None
        sch = get_scheduler(opt, args)

        if "DIFEX" in args.algorithm:
            ms = time.time()
            n_steps = args.max_epoch * args.steps_per_epoch
            print("start training fft teacher net")
            opt1 = get_optimizer(algorithm.teaNet, args, isteacher=True)
            sch1 = get_scheduler(opt1, args)
            algorithm.teanettrain(train_loaders, n_steps, opt1, sch1)
            print("complet time:%.4f" % (time.time() - ms))

        acc_record = {}
        acc_type_list = ["target"]
        train_minibatches_iterator = zip(*train_loaders)
        best_valid_acc, target_acc = 0, 0
        print("===========start training===========")
        sss = time.time()
        s = ""
        for _ in range(100):
            for item in acc_type_list:
                acc_record[item] = np.mean(
                    np.array(
                        [
                            modelopera.accuracy(algorithm, eval_loaders[i])
                            for i in eval_name_dict[item]
                        ]
                    )
                )
                s += item + "_acc:%.4f," % acc_record[item]
        print(args.algorithm)
        times += "%.2f " % (time.time() - sss)
    print(times)
