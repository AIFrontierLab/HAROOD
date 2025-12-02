# coding=utf-8
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader
import copy

from datautil.data_aug import raw_to_aug
import datautil.actdata.util as actutil
from datautil.mydataloader import InfiniteDataLoader

import datautil.actdata.cross_dataset as cross_dataset
import datautil.actdata.cross_people as cross_people
import datautil.actdata.cross_position as cross_position
import datautil.actdata.cross_time as cross_time
import datautil.actdata.cross_device as cross_device

task_act = {
    "cross_dataset": cross_dataset,
    "cross_people": cross_people,
    "cross_position": cross_position,
    "cross_time": cross_time,
    "cross_device": cross_device,
}


def split_trian_val_test(args, tmpdatay, rate):
    l = len(tmpdatay)
    if args.split_style == "strat":
        lslist = np.arange(l)
        stsplit = ms.StratifiedShuffleSplit(
            2, test_size=rate, train_size=1 - rate, random_state=args.seed
        )
        stsplit.get_n_splits(lslist, tmpdatay)
        indextr, indexte = next(stsplit.split(lslist, tmpdatay))
    else:
        indexall = np.arange(l)
        np.random.seed(args.seed)
        np.random.shuffle(indexall)
        ted = int(l * rate)
        indextr, indexte = indexall[:-ted], indexall[-ted:]
    return indextr, indexte


def get_dataloader(args, trdatalist, tedatalist):
    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=None,
            batch_size=args.batch_size,
            num_workers=args.N_WORKERS,
        )
        for env in trdatalist
    ]

    eval_loaders = [
        DataLoader(
            dataset=env,
            batch_size=64,
            num_workers=args.N_WORKERS,
            drop_last=False,
            shuffle=False,
        )
        for env in trdatalist + tedatalist
    ]
    return train_loaders, eval_loaders


def get_act_dataloader(args):
    pcross_act = task_act[args.task]
    rate = 0.2
    if args.task == "cross_people" or args.task == "cross_time":
        tmpp = args.act_people[args.dataset]
    elif args.task == "cross_position":
        tmpp = args.act_positon[args.dataset]
    elif args.task == "cross_device":
        tmpp = args.act_device[args.dataset]
    else:
        tmpp = args.act_dataset
    args.domain_num = len(tmpp)
    trdatalist, tedatalist = [], []
    for i, item in enumerate(tmpp):
        if i in args.test_envs:
            tedatalist.append(
                pcross_act.ActList(
                    args,
                    args.dataset,
                    args.data_dir,
                    item,
                    i,
                    transform=actutil.act_test(),
                )
            )
        else:
            tmpdatay = pcross_act.ActList(
                args,
                args.dataset,
                args.data_dir,
                item,
                i,
                transform=actutil.act_train(),
            ).labels
            indextr, indexte = split_trian_val_test(args, tmpdatay, rate)

            trdatalist.append(
                pcross_act.ActList(
                    args,
                    args.dataset,
                    args.data_dir,
                    item,
                    i,
                    transform=actutil.act_train(),
                    indices=indextr,
                )
            )
            tedatalist.append(
                pcross_act.ActList(
                    args,
                    args.dataset,
                    args.data_dir,
                    item,
                    i,
                    transform=actutil.act_test(),
                    indices=indexte,
                )
            )

    train_loaders, eval_loaders = get_dataloader(args, trdatalist, tedatalist)

    return train_loaders, eval_loaders


def get_aug_dataloaders(args, train_loaders):
    aug_datasets = []
    for item in train_loaders:
        tmpdataset = item.dataset
        tx, ty = tmpdataset.x, tmpdataset.labels
        cdataset = copy.deepcopy(tmpdataset)
        nx, ny, nd = raw_to_aug(tx, ty)
        cdataset.set_x(nx)
        cdataset.set_labels(ny, label_type="class_label")
        cdataset.set_labels(nd, label_type="domain_label")
        aug_datasets.append(cdataset)
    aug_train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=None,
            batch_size=args.batch_size,
            num_workers=args.N_WORKERS,
        )
        for env in aug_datasets
    ]
    return aug_train_loaders
