# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm


class LAG(Algorithm):
    def __init__(self, args, initx=None):
        super(LAG, self).__init__(args)

        self.featurizer = get_fea(args)
        self.featurizer.init(initx)
        self.bottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer
        )
        self.bottleneck2 = common_network.feat_bottleneck(
            self.featurizer.in_features1, args.bottleneck, args.layer
        )
        self.classifier = common_network.feat_classifier(
            args.num_classes, args.bottleneck * 2, args.classifier
        )

        self.args = args
        self.kernel_type = "mean_cov"

    def coral(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def update(self, minibatches, opt, sch):
        objective = 0
        penalty = 0
        penaltyr = 0
        nmb = len(minibatches)
        features = [self.featurizer(data[0].cuda().float()) for data in minibatches]
        feab1 = [self.bottleneck(item) for (item, _) in features]
        feab2 = [self.bottleneck2(item) for (_, item) in features]
        classifs = [
            self.classifier(torch.hstack((feab1[i], feab2[i])))
            for i in range(len(feab1))
        ]
        targets = [data[1].cuda().long() for data in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.coral(feab1[i], feab1[j])
                penaltyr += self.coral(feab2[i], feab2[j])

        objective /= nmb
        if nmb > 1:
            penalty /= nmb * (nmb - 1) / 2
            penaltyr /= nmb * (nmb - 1) / 2

        opt.zero_grad()
        (
            objective
            + (self.args.mmd_gamma * penalty)
            + (self.args.rela_gamma * penaltyr)
        ).backward()
        opt.step()
        if sch:
            sch.step()
        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {
            "class": objective.item(),
            "coral": penalty,
            "rela": penaltyr.item(),
            "total": (
                objective.item()
                + (self.args.mmd_gamma * penalty)
                + self.args.rela_gamma * penaltyr.item()
            ),
        }

    def extract_features(self, x):
        fea1, fea2 = self.featurizer(x)
        x1, x2 = self.bottleneck(fea1), self.bottleneck2(fea2)
        x = torch.hstack((x1, x2))
        return x

    def predict(self, x):
        fea1, fea2 = self.featurizer(x)
        x1, x2 = self.bottleneck(fea1), self.bottleneck2(fea2)
        x = torch.hstack((x1, x2))
        y = self.classifier(x)
        return y
