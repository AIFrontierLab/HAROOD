# coding=utf-8
import random
import numpy as np
import torch
import sys
import os
import torchvision
import PIL


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(filename, alg, args):
    save_dict = {"args": vars(args), "model_dict": alg.cpu().state_dict()}
    torch.save(save_dict, os.path.join(args.output, filename))
    alg.cuda()


def train_valid_target_eval_names(args):
    eval_name_dict = {"train": [], "valid": [], "target": []}
    t = 0
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict["train"].append(t)
            t += 1
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict["valid"].append(t)
        else:
            eval_name_dict["target"].append(t)
        t += 1
    return eval_name_dict


def get_str_from_args(args):
    if args.algorithm == "ERMPlusPlus":
        return f"{args.batch_size}_{args.lr}_{args.lars}_{args.linear_steps}"
    elif args.algorithm == "MLDG":
        # return f"{args.batch_size}_{args.lr}_{args.mldg_beta}_{args.inner_lr}"
        return f"{args.batch_size}_{args.lr}_{args.mldg_beta}"
    elif args.algorithm == "RSC":
        return f"{args.batch_size}_{args.lr}_{args.rsc_f_drop_factor}_{args.rsc_b_drop_factor}"
    elif args.algorithm == "Fishr":
        return f"{args.batch_size}_{args.lr}_{args.lam}_{args.penalty_anneal_iters}_{args.ema}"
    elif args.algorithm == "Fish":
        return f"{args.batch_size}_{args.lr}_{args.meta_lr}"
    elif args.algorithm == "VREx":
        return f"{args.batch_size}_{args.lr}_{args.lam}"
    elif args.algorithm == "ANDMask":
        return f"{args.batch_size}_{args.lr}_{args.tau}"
    elif args.algorithm == "Mixup":
        return f"{args.batch_size}_{args.lr}_{args.mixupalpha}"
    elif args.algorithm == "GroupDRO":
        return f"{args.batch_size}_{args.lr}_{args.groupdro_eta}"
    elif args.algorithm == "DANN":
        return f"{args.batch_size}_{args.lr}_{args.alpha}"
    elif args.algorithm == "URM":
        return f"{args.batch_size}_{args.lr}_{args.urm_adv_lambda}_{args.urm_discriminator_hidden_layers}_{args.urm_generator_output}_{args.urm_discriminator_label_smoothing}"
    elif args.algorithm == "CORAL":
        return f"{args.batch_size}_{args.lr}_{args.mmd_gamma}"
    elif args.algorithm == "LAG":
        return f"{args.batch_size}_{args.lr}_{args.mmd_gamma}_{args.rela_gamma}"
    elif args.algorithm == "DDLearn":
        return f"{args.batch_size}_{args.lr}_{args.auglossweight}_{args.dpweight}_{args.conweight}"
    elif args.algorithm == "MMD":
        return f"{args.batch_size}_{args.lr}_{args.mmd_gamma}"
    elif args.algorithm == "ERM":
        return f"{args.batch_size}_{args.lr}"
    return f"{args.batch_size}_{args.lr}_{args.alpha}_{args.beta}_{args.groupdro_eta}_{args.lam}_{args.mixupalpha}_{args.mldg_beta}_{args.mmd_gamma}_{args.rsc_f_drop_factor}_{args.rsc_b_drop_factor}_{args.tau}"


def alg_loss_dict(args):
    loss_dict = {
        "ANDMask": ["total"],
        "CORAL": ["class", "coral", "total"],
        "LAG": ["class", "coral", "rela", "total"],
        "DDLearn": ["class", "selfsup", "dp", "con", "total"],
        "DANN": ["class", "dis", "total"],
        "URM": ["total"],
        "ERM": ["class"],
        "ERMPlusPlus": ["class"],
        "Mixup": ["class"],
        "MLDG": ["total"],
        "MMD": ["class", "mmd", "total"],
        "GroupDRO": ["group"],
        "RSC": ["class"],
        "Fish": ["loss"],
        "Fishr": ["loss", "nll", "penalty"],
        "VREx": ["loss", "nll", "penalty"],
    }
    return loss_dict[args.algorithm]


def print_args(args, print_list):
    s = "==========================================\n"
    l = len(print_list)
    for arg, content in args.__dict__.items():
        if l == 0 or arg in print_list:
            s += "{}:{}\n".format(arg, content)
    return s


def print_environ():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def act_param_init(args):
    args.select_position = {
        "dsads": [0],
        "usc": [0],
        "har": [0],
        "pamap": [1],
        "emg": [0],
        "spcmd": [0],
        "wesad": [0],
    }
    args.select_channel = {
        "dsads": np.arange(6),
        "usc": np.arange(6),
        "har": np.arange(6),
        "pamap": np.arange(6),
        "emg": np.arange(8),
        "spcmd": np.arange(20),
        "wesad": np.arange(8),
    }
    args.label_cor = {
        "dsads": [[0], [1], [2, 3], [4], [5], [8]],
        "usc": [[7], [8], [9], [3], [4], [0]],
        "har": [[3], [4], [5], [1], [2], [0]],
        "pamap": [[1], [2], [0], [7], [8], [3]],
    }
    args.hz_list = {
        "dsads": 25,
        "usc": 100,
        "har": 50,
        "pamap": 100,
        "emg": 1000,
        "spcmd": 81,
        "wesad": 33,
        "opp": 30,
        "realworld": 50,
    }
    args.act_dataset = ["dsads", "usc", "har", "pamap"]
    args.act_people = {
        "dsads": [[i * 2, i * 2 + 1] for i in range(4)],
        "usc": [[1, 11, 2, 0], [6, 3, 9, 5], [7, 13, 8, 10], [4, 12]],
        "har": [[i * 6 + j for j in range(6)] for i in range(5)],
        "pamap": [[3, 2, 8], [1, 5], [0, 7], [4, 6]],
        "shar": [[0], [1], [2], [3]],
        "emg": [[i * 9 + j for j in range(9)] for i in range(4)],
        "shemg": [[i * 9 + j for j in range(9)] for i in range(4)],
        "loemg": [[i * 9 + j for j in range(9)] for i in range(4)],
        "spcmd": [[0]],
        "wesad": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14]],
        "opp": [[i] for i in range(4)],
        "realworld": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14]],
    }
    args.act_positon = {
        "dsads": [[i] for i in range(5)],
        "opp": [[i] for i in range(5)],
        "realworld": [[i] for i in range(7)],
        "usc": [[i] for i in range(1)],
        "har": [[i] for i in range(1)],
        "pamap": [[i] for i in range(3)],
    }
    args.act_device = {
        "dsads": [[0, 3], [3, 6], [6, 9]],
    }
    if args.task == "cross_dataset":
        args.num_classes = 6
        args.input_shape = (6, 1, 50)
        args.grid_size = 5
    else:
        if args.task == "cross_people" or args.task == "cross_time":
            tmp = {
                "dsads": ((45, 1, 125), 19, 5),
                "opp": ((45, 1, 200), 4, 10),
                "realworld": ((9, 1, 200), 8, 10),
                "usc": ((6, 1, 200), 12, 10),
                "har": ((6, 1, 128), 6, 8),
                "pamap": ((27, 1, 200), 12, 10),
                "shar": ((3, 1, 151), 17, 10),
                "emg": ((8, 1, 200), 6, 10),
                "shemg": ((8, 1, 100), 6, 10),
                "loemg": ((8, 1, 500), 6, 10),
                "spcmd": ((20, 1, 81), 10, 9),
                "wesad": ((8, 1, 200), 4, 10),
            }
        elif args.task == "cross_position":
            tmp = {
                "dsads": ((9, 1, 125), 19, 5),
                "usc": ((6, 1, 200), 12, 10),
                "har": ((6, 1, 128), 6, 8),
                "pamap": ((9, 1, 200), 12, 10),
            }
        elif args.task == "cross_device":
            tmp = {"dsads": ((15, 1, 125), 19, 5)}
        args.num_classes, args.input_shape, args.grid_size = (
            tmp[args.dataset][1],
            tmp[args.dataset][0],
            tmp[args.dataset][2],
        )

    return args
