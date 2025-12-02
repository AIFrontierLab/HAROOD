import os
import shlex

params_list = {
    "ERMPlusPlus": ["batch_size", "lr", "lars", "linear_steps"],
    "MLDG": ["batch_size", "lr", "mldg_beta"],
    "Fishr": ["batch_size", "lr", "lam", "penalty_anneal_iters", "ema"],
    "Fish": ["batch_size", "lr", "meta_lr"],
    "RSC": ["batch_size", "lr", "rsc_f_drop_factor", "rsc_b_drop_factor"],
    "VREx": ["batch_size", "lr", "lam"],
    "ANDMask": ["batch_size", "lr", "tau"],
    "Mixup": ["batch_size", "lr", "mixupalpha"],
    "GroupDRO": ["batch_size", "lr", "groupdro_eta"],
    "DANN": ["batch_size", "lr", "alpha"],
    "URM": [
        "batch_size",
        "lr",
        "urm_adv_lambda",
        "urm_discriminator_hidden_layers",
        "urm_generator_output",
        "urm_discriminator_label_smoothing",
    ],
    "CORAL": ["batch_size", "lr", "mmd_gamma"],
    "LAG": ["batch_size", "lr", "mmd_gamma", "rela_gamma"],
    "DDLearn": ["batch_size", "lr", "auglossweight", "dpweight", "conweight"],
    "MMD": ["batch_size", "lr", "mmd_gamma"],
    "ERM": ["batch_size", "lr"],
}

fixed_params_list = {"seed": [0, 42, 96]}


def parse_script(script):
    tokens = shlex.split(script)
    args = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                args[token.lstrip("--")] = tokens[i + 1]
                i += 2
            else:
                args[token.lstrip("--")] = True
                i += 1
        else:
            i += 1
    return args


def get_str_from_args(args):
    filename = args.algorithm.lower()
    filename = os.path.join("./scripts/tmp", filename + ".txt")
    with open(filename, "r") as f:
        scripts = f.read().split("\n")
    datas = []
    keys = params_list[args.algorithm]
    for item in scripts:
        if len(item) > 0:
            tdatas = parse_script(item)
            if (
                tdatas["task"] == args.task
                and tdatas["dataset"] == args.dataset
                and tdatas["test_envs"] == str(args.test_envs[0])
            ):
                ts = f"{tdatas['seed']}/"
                for key in keys:
                    if key in tdatas:
                        ts += f"{tdatas[key]}_"
                datas.append(ts[:-1])
    return datas


def get_valid_acc(filepath):
    with open(os.path.join(filepath, "done.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            if "valid acc" in line:
                return float(line.split(":")[-1].strip())


def get_valid_f1(filepath):
    with open(os.path.join(filepath, "f1done.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            if "valid f1" in line:
                return float(line.split(":")[-1].strip())


def get_target_f1(filepath):
    with open(os.path.join(filepath, "f1done.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            if "target f1" in line:
                return float(line.split(":")[-1].strip())


def get_target_acc(filepath):
    with open(os.path.join(filepath, "done.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            if "target acc" in line:
                return float(line.split(":")[-1].strip())
