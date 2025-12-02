# coding=utf-8
import numpy as np
import torch.nn.functional as F
import torch
from alg.algs.ERM import ERM


class Mixup(ERM):
    def __init__(self, args):
        super(Mixup, self).__init__(args)
        self.args = args

    def update(self, minibatches, opt, sch):

        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        indexes1 = torch.randperm(len(all_x))
        lam = np.random.beta(self.args.mixupalpha, self.args.mixupalpha)
        nx = all_x * lam + all_x[indexes1] * (1 - lam)
        pre = self.predict(nx)
        objective = 0
        objective += lam * F.cross_entropy(pre, all_y)
        objective += (1 - lam) * F.cross_entropy(pre, all_y[indexes1])

        opt.zero_grad()
        objective.backward()
        opt.step()
        if sch:
            sch.step()
        return {"class": objective.item()}
