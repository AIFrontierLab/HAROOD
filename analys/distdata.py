import torch
import numpy as np
import argparse
import ot
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

# python -m analys.distdata --data_dir ./data/ --task cross_people --dataset dsads --seed 42
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
            ).x
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
        default="img_dg",
        choices=[
            "img_dg",
            "cross_dataset",
            "cross_people",
            "cross_position",
            "cross_time",
        ],
        help="now only support image tasks",
    )
    args = parser.parse_args()
    args.steps_per_epoch = 100
    args.data_dir = args.data_file + args.data_dir
    args.test_envs = [100]
    args = act_param_init(args)
    return args


def wasserstein_1d(x, y):
    return wasserstein_distance(x, y)


def my_cdist(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(
        x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
    ).add_(x1_norm)
    return res.clamp_min_(1e-30)


def gaussian_kernel(x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
    D = my_cdist(x, y)
    K = torch.zeros_like(D)

    for g in gamma:
        K.add_(torch.exp(D.mul(-g)))

    return K


def mmd_rbf(x, y):
    Kxx = gaussian_kernel(x, x).mean()
    Kyy = gaussian_kernel(y, y).mean()
    Kxy = gaussian_kernel(x, y).mean()
    return Kxx + Kyy - 2 * Kxy


def normalize_domain(x):
    N, C, _, L = x.shape
    x_flat = x.reshape(N, C * L)
    mu = x_flat.mean(axis=0, keepdims=True)
    sigma = x_flat.std(axis=0, keepdims=True) + 1e-6
    x_norm = (x_flat - mu) / sigma
    return x_norm


def wasserstein_multid(x, y, metric="euclidean"):
    a = np.ones((x.shape[0],)) / x.shape[0]
    b = np.ones((y.shape[0],)) / y.shape[0]
    M = ot.dist(x, y, metric=metric)
    return ot.emd2(a, b, M)


def compute_domain_distances(domains):
    D = len(domains)
    mmd_mat = np.zeros((D, D))
    w1_mat = np.zeros((D, D))
    W_mat = np.zeros((D, D))
    for i in range(D):
        for j in range(i + 1, D):
            xi = normalize_domain(domains[i].numpy())  # shape (N, C, 1, L)
            xj = normalize_domain(domains[j].numpy())

            xi_t = torch.from_numpy(xi).float()
            xj_t = torch.from_numpy(xj).float()

            m = mmd_rbf(xi_t, xj_t).item()
            mmd_mat[i, j] = mmd_mat[j, i] = m

            mean_i = xi.mean(axis=1)
            mean_j = xj.mean(axis=1)
            w = wasserstein_1d(mean_i, mean_j)
            w1_mat[i, j] = w1_mat[j, i] = w

            W_mat[i, j] = W_mat[j, i] = wasserstein_multid(xi, xj)
    return mmd_mat, w1_mat, W_mat


def print_matrix(x):
    s = ""
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            s += "{:.2f} ".format(x[i, j])
        s += "\n"
    print(s)


args = get_args()
domains = get_act_datax(args)
mmd_mat, w1_mat, W_mat = compute_domain_distances(domains)
print(args.task, args.dataset)
print_matrix(mmd_mat)
print_matrix(w1_mat)
print_matrix(W_mat)
print(np.mean(np.array(mmd_mat)), np.mean(np.array(w1_mat)), np.mean(np.array(W_mat)))
