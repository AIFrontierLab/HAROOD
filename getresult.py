import argparse
import os
import numpy as np
from utils.util import act_param_init
from postprocess.utils import (
    fixed_params_list,
    params_list,
    parse_script,
    get_valid_acc,
    get_target_acc,
)

parser = argparse.ArgumentParser(description="Domain generalization")
parser.add_argument("--algorithm", type=str, nargs="+", default=["ERM"])
parser.add_argument("--dataset", type=str, default="dsads")
parser.add_argument("--task", type=str, default="cross_people")
parser.add_argument("--net", type=str, default="cnn")
parser.add_argument("--output", type=str, default="/data_2/lw/DeepDG/output/")
parser.add_argument("--selstyle", type=str, default="valid", choices=["valid", "test"])
args = parser.parse_args()
args = act_param_init(args)

if args.task == "cross_people":
    domain_num = len(args.act_people[args.dataset])
elif args.task == "cross_position":
    domain_num = len(args.act_positon[args.dataset])
else:
    domain_num = 4

resrec = np.zeros((domain_num, len(fixed_params_list["seed"])))

# python -m getresult --net cnn --task cross_people --dataset dsads --algorithm ERM CORAL MMD DANN SimMixup RSC MLDG ANDMask VREx GroupDRO Fish Fishr URM ERMPlusPlus SMixup


for alg in args.algorithm:
    for i in range(domain_num):
        args.test_envs = [i]
        tmpacc = np.zeros((len(fixed_params_list["seed"]), 20))
        tmpfile = []
        for j, seed in enumerate(fixed_params_list["seed"]):
            args.seed = int(seed)
            if args.net == "cnn":
                tdir = os.path.join(
                    args.output, args.task, args.dataset, str(i), alg, str(args.seed)
                )
            else:
                tdir = os.path.join(
                    args.output,
                    args.net,
                    args.task,
                    args.dataset,
                    str(i),
                    alg,
                    str(args.seed),
                )
            if j == 0:
                files = os.listdir(tdir)
                files = [os.path.join(tdir, item) for item in files]
            else:
                files = tmpfile
            tind = 0
            for k, item in enumerate(files):
                if j == 0:
                    if os.path.exists(os.path.join(item, "done.txt")):
                        tmpfile.append(item)
                        if args.selstyle == "valid":
                            tmpacc[j, tind] = get_valid_acc(item)
                        elif args.selstyle == "test":
                            tmpacc[j, tind] = get_target_acc(item)
                        tind += 1
                else:
                    parts = item.split(os.sep)
                    parts[-2] = str(seed)
                    new_item = os.sep.join(parts)
                    if args.selstyle == "valid":
                        tmpacc[j, k] = get_valid_acc(new_item)
                    elif args.selstyle == "test":
                        tmpacc[j, k] = get_target_acc(new_item)
        tmpaccmean = np.mean(tmpacc * 100, axis=0)
        flagfile = tmpfile[np.argmax(tmpaccmean)]
        for j, seed in enumerate(fixed_params_list["seed"]):
            parts = flagfile.split(os.sep)
            parts[-2] = str(seed)
            new_flagfile = os.sep.join(parts)
            resrec[i, j] = get_target_acc(new_flagfile) * 100
    meanres = np.mean(resrec, axis=1)
    stdres = np.std(resrec, axis=1)
    s = ""
    s += "%s " % alg
    for i in range(domain_num):
        s += "%.2f±%.2f " % (meanres[i], stdres[i])

    tmean = np.mean(resrec, axis=0)
    overall_mean = np.mean(tmean)
    overall_std = np.std(tmean)
    s += "%.2f±%.2f" % (overall_mean, overall_std)
    print(s)
