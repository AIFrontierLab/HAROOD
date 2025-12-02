# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm
from alg.opt import ErmPlusPlusMovingAvg, LARS


class ERMPlusPlus(Algorithm, ErmPlusPlusMovingAvg):
    def __init__(self, args):
        Algorithm.__init__(self, args)
        self.args = args
        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier(
            args.num_classes, self.featurizer.in_features, args.classifier
        )

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.network = self.network.cuda()
        if args.lars:
            self.optimizer = LARS(
                self.network.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                foreach=False,
            )
        linear_parameters = []
        for n, p in self.network[1].named_parameters():
            linear_parameters.append(p)

        if args.lars:
            self.linear_optimizer = LARS(
                linear_parameters, lr=args.lr, weight_decay=args.weight_decay
            )
        else:
            self.linear_optimizer = torch.optim.Adam(
                linear_parameters,
                lr=args.lr,
                weight_decay=args.weight_decay,
                foreach=False,
            )
        self.lr_schedule = []
        self.lr_schedule_changes = 0
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", patience=1
        )
        ErmPlusPlusMovingAvg.__init__(self, self.network)

    def update(self, minibatches, opt, sch):
        if self.global_iter > self.args.linear_steps:
            selected_optimizer = self.optimizer
        else:
            selected_optimizer = self.linear_optimizer

        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        loss = F.cross_entropy(self.network(all_x), all_y)

        selected_optimizer.zero_grad()
        loss.backward()
        selected_optimizer.step()
        self.update_sma()

        return {"class": loss.item()}

    def predict(self, x):
        self.network_sma.eval()
        return self.network_sma(x)
