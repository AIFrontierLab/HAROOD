# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import copy

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm
from alg.opt import ParamDict


class WholeFish(nn.Module):
    def __init__(self, args, weights=None):
        super(WholeFish, self).__init__()
        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier(
            args.num_classes, self.featurizer.in_features, args.classifier
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)

        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.network(x)


class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, args):
        super(Fish, self).__init__(args)
        self.args = args
        self.network = WholeFish(args)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = WholeFish(self.args, weights=self.network.state_dict()).to(
            device
        )
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, opt, sch):
        self.create_clone("cuda")

        for data in minibatches:
            loss = F.cross_entropy(
                self.network_inner(data[0].cuda().float()), data[1].cuda().long()
            )
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.args.meta_lr,
        )
        self.network.reset_weights(meta_weights)

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)
