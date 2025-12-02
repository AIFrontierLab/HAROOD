# coding=utf-8

from datautil.actdata.util import *
from datautil.util import Nmax
import numpy as np


class ActList(object):
    def __init__(
        self,
        args,
        dataset,
        root_dir,
        device_group,
        group_num,
        transform=None,
        target_transform=None,
        indices=None,
        pclabels=None,
        pdlabels=None,
    ):
        self.domain_num = 0
        self.dataset = dataset
        self.task = "cross-device"
        self.transform = transform
        self.target_transform = target_transform
        x, cy, py, sy = loaddata_from_numpy(self.dataset, self.task, root_dir)
        self.device_group = device_group
        self.people = np.sort(np.unique(py))
        self.position = np.sort(np.unique(sy))
        self.comb_position(x, cy, py, sy)
        self.x = self.x[:, :, np.newaxis, :]
        self.transform = None
        self.x = torch.tensor(self.x).float()
        if pclabels is not None:
            self.pclabels = pclabels
        else:
            self.pclabels = np.ones(self.labels.shape) * (-1)
        if pdlabels is not None:
            self.pdlabels = pdlabels
        else:
            self.pdlabels = np.ones(self.labels.shape) * (0)
        self.tdlabels = np.ones(self.labels.shape) * group_num
        self.dlabels = np.ones(self.labels.shape) * (group_num - Nmax(args, group_num))
        if indices is None:
            self.indices = np.arange(len(self.x))
        else:
            self.indices = indices

    def comb_position(self, x, cy, py, sy):
        for i, peo in enumerate(self.people):
            index = np.where(py == peo)[0]
            tx, tcy, tsy = x[index], cy[index], sy[index]
            for j, sen in enumerate(self.position):
                index = np.where(tsy == sen)[0]
                if j == 0:
                    ttx, ttcy = (
                        tx[index, self.device_group[0] : self.device_group[1]],
                        tcy[index],
                    )
                else:
                    ttx = np.hstack(
                        (ttx, tx[index, self.device_group[0] : self.device_group[1]])
                    )
            if i == 0:
                self.x, self.labels = ttx, ttcy
            else:
                self.x, self.labels = np.vstack((self.x, ttx)), np.hstack(
                    (self.labels, ttcy)
                )

    def set_labels(self, tlabels=None, label_type="domain_label"):
        assert len(tlabels) == len(self.x)
        if label_type == "domain_label":
            self.dlabels = tlabels
        elif label_type == "class_label":
            self.labels = tlabels

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        index = self.indices[index]
        img = self.input_trans(self.x[index])
        ctarget = self.target_trans(self.labels[index])
        dtarget = self.target_trans(self.dlabels[index])
        return img, ctarget, dtarget

    def set_x(self, x):
        self.x = x

    def __len__(self):
        return len(self.indices)
