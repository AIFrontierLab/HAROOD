# coding=utf-8

import os
import sys
import time
import numpy as np
from argparse import Namespace
from typing import Union, Dict, Any
import yaml

from alg.opt import *
from alg import alg, modelopera
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


DEFAULT_ARGS = {
    "algorithm": "ERM",
    "alpha": 1.0,
    "anneal_iters": 500,
    "batch_size": 32,
    "beta": 1.0,
    "beta1": 0.5,
    "linear_steps": 500,
    "lars": False,
    "bottleneck": 256,
    "checkpoint_freq": 3,
    "classifier": "linear",
    "data_file": "",
    "dataset": "dsads",
    "data_dir": "../DeepDG/data/",
    "dis_hidden": 256,
    "disttype": "2-norm",
    "distyle": "l1",
    "urm_discriminator_hidden_layers": 2,
    "urm_generator_output": "tanh",
    "urm_adv_lambda": 0.1,
    "urm_discriminator_label_smoothing": 0,
    "gpu_id": "0",
    "groupdro_eta": 1.0,
    "inner_lr": 1e-2,
    "lam": 1.0,
    "layer": "bn",
    "lr": 1e-2,
    "meta_lr": 1e-1,
    "lr_decay": 0.75,
    "lr_decay1": 1.0,
    "lr_decay2": 1.0,
    "lr_gamma": 0.0003,
    "max_epoch": 120,
    "penalty_anneal_iters": 1500,
    "ema": 0.95,
    "mixupalpha": 0.2,
    "mldg_beta": 1.0,
    "mmd_gamma": 1.0,
    "rela_gamma": 1.0,
    "auglossweight": 1.0,
    "dpweight": 1.0,
    "conweight": 1.0,
    "momentum": 0.9,
    "net": "resnet50",
    "N_WORKERS": 1,
    "rsc_f_drop_factor": 1 / 3,
    "rsc_b_drop_factor": 1 / 3,
    "save_model_every_checkpoint": False,
    "schuse": False,
    "schusech": "cos",
    "model_size": "medium",
    "seed": 0,
    "split_style": "strat",
    "task": "cross_people",
    "tau": 1.0,
    "test_envs": [0],
    "output": "train_output",
    "weight_decay": 5e-4,
}


def load_config(config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(config, str) and os.path.exists(config):
        with open(config, "r") as f:
            return yaml.safe_load(f)
    elif isinstance(config, dict):
        return config
    else:
        return {}


def merge_args(default: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Namespace:
    merged = default.copy()
    merged.update(config)
    merged.update(kwargs)

    if "test_envs" in merged and isinstance(merged["test_envs"], str):
        merged["test_envs"] = [int(x) for x in merged["test_envs"].split(",")]

    return Namespace(**merged)


def train(config: Union[str, Dict[str, Any]] = None, **kwargs) -> Dict[str, float]:
    torch.cuda.reset_peak_memory_stats()
    config_dict = load_config(config)
    args = merge_args(DEFAULT_ARGS, config_dict, **kwargs)

    args.steps_per_epoch = 100
    args.data_dir = args.data_file + args.data_dir
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

    if os.path.exists(os.path.join(args.output, "done.txt")):
        print("already done")
        return {}

    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, "out.txt"))
    sys.stderr = Tee(os.path.join(args.output, "err.txt"))

    args = act_param_init(args)

    print_environ()
    loss_list = alg_loss_dict(args)
    train_loaders, eval_loaders = get_act_dataloader(args)

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
    s = print_args(args, [])
    print("=======hyper-parameter used========")
    print(s)

    acc_record = {}
    acc_type_list = ["train", "valid", "target"]
    train_minibatches_iterator = zip(*train_loaders)
    best_valid_acc, target_acc = 0, 0
    print("===========start training===========")
    sss = time.time()

    for epoch in range(args.max_epoch):
        for iter_num in range(args.steps_per_epoch):
            minibatches_device = [(data) for data in next(train_minibatches_iterator)]

            if args.algorithm == "VREx" and algorithm.update_count == args.anneal_iters:
                opt = get_optimizer(algorithm, args)
                sch = get_scheduler(opt, args)

            if args.algorithm == "DDLearn":
                aug_minibatches_device = [
                    (data) for data in next(aug_train_minibatches_iterator)
                ]
                step_vals = algorithm.update(
                    minibatches_device, aug_minibatches_device, opt, sch
                )
            else:
                step_vals = algorithm.update(minibatches_device, opt, sch)

        if (
            not args.algorithm in ["Fishr", "Fish"]
            and (epoch in [int(args.max_epoch * 0.7), int(args.max_epoch * 0.9)])
            and not args.schuse
            and not args.task.startswith("cross")
        ):
            print("manually descrease lr")
            for params in opt.param_groups:
                params["lr"] = params["lr"] * 0.1

        if (epoch == (args.max_epoch - 1)) or (epoch % args.checkpoint_freq == 0):
            print("===========epoch %d===========" % (epoch))
            s = ""
            for item in loss_list:
                s += item + "_loss:%.4f," % step_vals[item]
            print(s[:-1])
            s = ""

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
            print(s[:-1])

            if acc_record["valid"] > best_valid_acc:
                best_valid_acc = acc_record["valid"]
                target_acc = acc_record["target"]

            if args.save_model_every_checkpoint:
                save_checkpoint(f"model_epoch{epoch}.pkl", algorithm, args)

            print("total cost time: %.4f" % (time.time() - sss))
            algorithm_dict = algorithm.state_dict()

    save_checkpoint("model.pkl", algorithm, args)
    print("valid acc: %.4f" % best_valid_acc)
    print("DG result: %.4f" % target_acc)

    with open(os.path.join(args.output, "done.txt"), "w") as f:
        f.write("done\n")
        f.write("total cost time:%s\n" % (str(time.time() - sss)))
        f.write("valid acc:%.4f\n" % (best_valid_acc))
        f.write("target acc:%.4f" % (target_acc))

    peak_bytes = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    return {
        "valid_acc": best_valid_acc,
        "target_acc": target_acc,
        "total_time": str(time.time() - sss),
        "peak_bytes": f"Peak memory: {peak_bytes / 1024**2:.2f} MB",
        "peak_reserved": f"Peak reserved memory: {peak_reserved / 1024**2:.2f} MB",
    }
