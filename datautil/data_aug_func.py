# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding: utf-8

# # Data augmentation for time-series data

# #### This is a simple example to apply data augmentation to time-series data (e.g. wearable sensor data). If it helps your research, please cite the below paper.

# T. T. Um et al., “Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks,” in Proceedings of the 19th ACM International Conference on Multimodal Interaction, ser. ICMI 2017. New York, NY, USA: ACM, 2017, pp. 216–220.

# https://dl.acm.org/citation.cfm?id=3136817
#
# https://arxiv.org/abs/1706.00527

# @inproceedings{TerryUm_ICMI2017,
#  author = {Um, Terry T. and Pfister, Franz M. J. and Pichler, Daniel and Endo, Satoshi and Lang, Muriel and Hirche, Sandra and Fietzek, Urban and Kuli\'{c}, Dana},
#  title = {Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring Using Convolutional Neural Networks},
#  booktitle = {Proceedings of the 19th ACM International Conference on Multimodal Interaction},
#  series = {ICMI 2017},
#  year = {2017},
#  isbn = {978-1-4503-5543-8},
#  location = {Glasgow, UK},
#  pages = {216--220},
#  numpages = {5},
#  doi = {10.1145/3136755.3136817},
#  acmid = {3136817},
#  publisher = {ACM},
#  address = {New York, NY, USA},
#  keywords = {Parkinson\&\#39;s disease, convolutional neural networks, data augmentation, health monitoring, motor state detection, wearable sensor},
# }

# #### You can freely modify this code for your own purpose. However, please leave the above citation information untouched when you redistributed the code to others. Please contact me via email if you have any questions. Your contributions on the code are always welcome. Thank you.

# Terry Taewoong Um (terry.t.um@gmail.com)
#
# https://twitter.com/TerryUm_ML
#
# https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data

import numpy as np
from scipy.interpolate import CubicSpline
from transforms3d.axangles import axangle2mat


def DA_Jitter(X, sigma=0.05):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + myNoise


def DA_Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0], 1))
    return X * scalingFactor


def GenerateRandomCurves(X, sigma=0.2, knot=4):
    C, L = X.shape
    xx = np.linspace(0, L - 1, num=knot + 2)
    yy = np.random.normal(loc=1.0, scale=sigma, size=(C, knot + 2))
    x_range = np.arange(L)
    curves = np.zeros((C, L))
    for c in range(C):
        cs = CubicSpline(xx, yy[c])
        curves[c] = cs(x_range)
    return curves


def DA_MagWarp(X, sigma):
    return X * GenerateRandomCurves(X, sigma)


def DistortTimesteps(X, sigma=0.2):
    C, L = X.shape
    curves = GenerateRandomCurves(X, sigma)
    tt_cum = np.cumsum(curves, axis=1)
    t_scale = (L - 1) / tt_cum[:, -1]
    tt_cum = tt_cum * t_scale[:, None]
    return tt_cum


def DA_TimeWarp(X, sigma=0.2):
    C, L = X.shape
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros_like(X)
    x_range = np.arange(L)
    for c in range(C):
        X_new[c] = np.interp(x_range, tt_new[c], X[c])
    return X_new


def DA_Rotation(X):
    C, L = X.shape
    if C == 3:
        axis = np.random.uniform(low=-1, high=1, size=3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        R = axangle2mat(axis, angle)
    else:
        Q, _ = np.linalg.qr(np.random.randn(C, C))
        R = Q
    X_rot = np.dot(R, X)
    return X_rot


def DA_Permutation(X, nPerm=4, minSegLength=10):
    C, L = X.shape
    X_new = np.zeros_like(X)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile:
        segs = np.zeros(nPerm + 1, dtype=int)
        segs[1:-1] = np.sort(
            np.random.randint(minSegLength, L - minSegLength, nPerm - 1)
        )
        segs[-1] = L
        if np.min(segs[1:] - segs[:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[:, segs[idx[ii]] : segs[idx[ii] + 1]]
        X_new[:, pp : pp + x_temp.shape[1]] = x_temp
        pp += x_temp.shape[1]
    return X_new


def RandSampleTimesteps(X, nSample):
    C, L = X.shape
    tt = np.zeros((C, nSample), dtype=int)
    for c in range(C):
        tt_c = np.sort(np.random.randint(1, L - 1, nSample - 2))
        tt[c, 1:-1] = tt_c
        tt[c, -1] = L - 1
    return tt


def DA_RandSampling(X, nSample_rate):
    C, L = X.shape
    nSample = int(L * nSample_rate)
    tt = RandSampleTimesteps(X, nSample)
    X_new = np.zeros_like(X)
    x_range = np.arange(L)
    for c in range(C):
        X_new[c] = np.interp(x_range, tt[c], X[c, tt[c]])
    return X_new
