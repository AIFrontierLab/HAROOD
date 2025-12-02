# coding=utf-8
import numpy as np
from datautil.data_aug_func import (
    DA_Jitter,
    DA_Scaling,
    DA_MagWarp,
    DA_TimeWarp,
    DA_Rotation,
    DA_Permutation,
    DA_RandSampling,
)


def raw_to_aug(x, y, aug_num=7, num_workers=4):
    N, C, _, L = x.shape
    x2 = x[:, :, 0, :]
    aug_funcs = [
        lambda x: DA_Jitter(x, sigma=0.05),
        lambda x: DA_Scaling(x, sigma=0.1),
        lambda x: DA_MagWarp(x, sigma=0.2),
        lambda x: DA_TimeWarp(x, sigma=0.2),
        lambda x: DA_Rotation(x),
        lambda x: DA_Permutation(x, nPerm=4, minSegLength=10),
        lambda x: DA_RandSampling(x, nSample_rate=0.4),
    ]
    aug_all_x = []
    for func in aug_funcs:
        batch_aug = np.stack([func(x2[i]) for i in range(N)], axis=0)  #
        aug_all_x.append(batch_aug[:, :, np.newaxis, :])
    aug_all_x = np.concatenate(aug_all_x, axis=0)
    aug_all_y = np.tile(y, aug_num)
    aug_label = np.repeat(np.arange(aug_num), N)
    return aug_all_x, aug_all_y, aug_label
