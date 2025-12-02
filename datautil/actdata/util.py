# coding=utf-8
from torchvision import transforms
import numpy as np
import math
import torch


def split_trian_val_test(da, rate=0.8, seed=0):
    dsize = len(da)
    tr_size = int(rate * dsize)
    tr_da, te_da = torch.utils.data.random_split(
        da, [tr_size, dsize - tr_size], generator=torch.Generator().manual_seed(seed)
    )
    return tr_da, te_da


def act_train():
    return transforms.Compose([transforms.ToTensor()])


def act_test():
    return transforms.Compose([transforms.ToTensor()])


def loaddata_from_numpy(dataset="dsads", task="cross_people", root_dir="./data/act/"):
    if dataset == "pamap" and task == "cross_people":
        x = np.load(root_dir + dataset + "/" + dataset + "_x1.npy")
        ty = np.load(root_dir + dataset + "/" + dataset + "_y1.npy")
    else:
        x = np.load(root_dir + dataset + "/" + dataset + "_x.npy")
        ty = np.load(root_dir + dataset + "/" + dataset + "_y.npy")
    cy, py, sy = ty[:, 0], ty[:, 1], ty[:, 2]
    return x, cy, py, sy


def loadxtdata_from_numpy(dataset="emg", root_dir="./data/xtime/"):
    x = np.load(root_dir + dataset + "/" + dataset + "_x.npy")
    ty = np.load(root_dir + dataset + "/" + dataset + "_y.npy")
    cy, dc = ty[:, 0], ty[:, 1]
    return x, cy, dc


def seq_cut(x, tl):
    tl = int(tl)
    l = x.shape[-1]
    dl = l - tl
    s = dl // 2
    if dl % 2 == 0:
        x = x[:, :, s : l - s]
    else:
        x = x[:, :, s : l - s - 1]
    return x


def split_via_wind_1(x, cy, sl, step):
    l = x.shape[-1]
    tl = l - l % sl
    tx, tcy = [], []
    for i in range(len(x)):
        start = 0
        end = sl
        while end <= l:
            tx.append(x[i, :, start:end])
            tcy.append(cy[i])
            start += step
            end += step
    return np.array(tx), np.array(tcy)


def seq_downsapmle(x, ohz, thz):
    t = int(ohz * thz / math.gcd(ohz, thz))
    do = int(t / ohz)
    to = int(t / thz)
    l = x.shape[-1]
    x = seq_cut(x, l - l % to)
    l = x.shape[-1]
    d = int(l / to)
    x = np.concatenate(
        [
            np.array(
                x[
                    :,
                    :,
                    np.arange(to * i, to * i + to)[
                        np.random.choice(np.arange(to), do, replace=False)
                    ],
                ]
            )
            for i in range(d)
        ],
        axis=-1,
    )
    return x
