from datautil.actdata.util import *
from datautil.util import Nmax
import numpy as np


class ActList(object):
    def __init__(
        self,
        args,
        datasetgroup,
        root_dir,
        dataset,
        domain_num,
        hz=25,
        win=2,
        transform=None,
        target_transform=None,
        indices=None,
        pclabels=None,
        pdlabels=None,
    ):
        self.domain_num = 0
        self.hz = hz
        self.win = win
        self.dataset = dataset
        self.task = "cross_dataset"
        self.transform = transform
        self.target_transform = target_transform
        x, cy, _, sy = loaddata_from_numpy(self.dataset, self.task, root_dir)
        x, cy = self.select_position_channel(args, x, cy, sy)
        cy = self.map_label(args, cy)
        index = np.where(cy >= 0)[0]
        x, cy = x[index], cy[index]
        x = seq_downsapmle(x, args.hz_list[self.dataset], hz)
        self.x, self.labels = split_via_wind_1(x, cy, win * hz, int(win * hz / 2))
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
        if indices is None:
            self.indices = np.arange(len(self.x))
        else:
            self.indices = indices
        self.tdlabels = np.ones(self.labels.shape) * domain_num
        self.dlabels = np.ones(self.labels.shape) * (
            domain_num - Nmax(args, domain_num)
        )

    def select_position_channel(self, args, x, cy, sy):
        sel_sen = args.select_position[self.dataset]
        sel_chn = np.array(args.select_channel[self.dataset])
        index = []
        for item in sel_sen:
            index.append(np.where(sy == item)[0])
        index = np.hstack(index)
        return x[index][:, sel_chn, :], cy[index]

    def map_label(self, args, cy):
        map_l = args.label_cor[self.dataset]
        tcy = np.ones(cy.shape) * (-1)
        for i, item in enumerate(map_l):
            for c in item:
                index = np.where(cy == c)[0]
                tcy[index] = i
        return tcy

    def set_labels(self, tlabels=None, label_type="domain_label"):
        assert len(tlabels) == len(self.x)
        if label_type == "domain_label":
            self.dlabels = tlabels
        elif label_type == "class_label":
            self.labels = tlabels

    def set_x(self, x):
        self.x = x

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
