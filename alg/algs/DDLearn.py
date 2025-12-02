# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm


class DDLearn(Algorithm):
    def __init__(self, args, n_aug_class=8):
        super(DDLearn, self).__init__(args)
        self.n_aug_class = n_aug_class
        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier(
            args.num_classes, self.featurizer.in_features, args.classifier
        )
        self.aug_classifier = common_network.feat_classifier(
            self.n_aug_class, self.featurizer.in_features, args.classifier
        )
        self.con = SupConLoss_m(contrast_mode="all")
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.args = args

    def update(self, minibatches, aug_minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        all_d = torch.ones_like(all_y) * 7
        au_x = torch.cat([data[0].cuda().float() for data in aug_minibatches])
        au_y = torch.cat([data[1].cuda().long() for data in aug_minibatches])
        au_d = torch.cat([data[2].cuda().long() for data in aug_minibatches])
        feature_ori = self.featurizer(all_x)
        auglabel_true = torch.cat((all_d, au_d), dim=0)
        feature_aug = self.featurizer(au_x)
        feature_aug_task = torch.cat((feature_ori, feature_aug), dim=0)
        auglabel_p = self.predict_aug(feature_aug_task)
        feature_act_task = feature_aug_task
        actlabel_true = torch.cat((all_y, au_y), dim=0)
        actlabel_p = self.predict_act(feature_act_task)

        loss_c = F.cross_entropy(actlabel_p, actlabel_true)

        loss_selfsup = F.cross_entropy(auglabel_p, auglabel_true)
        loss_dp = torch.zeros(1).cuda()
        dp_layer = DPLoss(input_dim=self.featurizer.in_features)
        loss_dp = dp_layer.compute(feature_ori, feature_aug)

        con_loss = self.con(
            torch.cat([feature_ori.unsqueeze(1), feature_aug.unsqueeze(1)], dim=1),
            torch.cat([all_y, au_y]),
        )
        loss = (
            loss_c
            + self.args.auglossweight * loss_selfsup
            + self.args.dpweight * loss_dp
            + self.args.conweight * con_loss
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {
            "class": loss_c.item(),
            "selfsup": loss_selfsup.item(),
            "dp": loss_dp.item(),
            "con": con_loss,
            "total": loss.item(),
        }

    def predict(self, x):
        return self.network(x)

    def predict_act(self, feature):
        act_predict = self.classifier(feature)
        return act_predict

    def predict_aug(self, feature):
        aug_predict = self.aug_classifier(feature)
        return aug_predict


class DPLoss(object):
    def __init__(self, loss_type="dis", input_dim=512):
        self.loss_type = loss_type
        self.input_dim = input_dim

    def compute(self, X, Y):
        if self.loss_type == "dis":
            loss = dis(X, Y, input_dim=self.input_dim, hidden_dim=60)
        return loss


class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, out_dim=1):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dis1 = nn.Linear(input_dim, hidden_dim)
        self.dis2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.dis1(x))
        x = self.dis2(x)
        x = torch.sigmoid(x)
        return x


def dis(source, target, input_dim=256, hidden_dim=512, dis_net=None):
    """discrimination loss for two domains.
    source and target are features.
    """
    domain_loss = nn.BCELoss()
    dis_net = Discriminator(input_dim, hidden_dim).cuda()
    domain_src = torch.ones(len(source)).cuda()
    domain_tar = torch.zeros(len(target)).cuda()
    domain_src, domain_tar = domain_src.view(domain_src.shape[0], 1), domain_tar.view(
        domain_tar.shape[0], 1
    )
    pred_src = dis_net(source)
    pred_tar = dis_net(target)
    loss_s, loss_t = domain_loss(pred_src, domain_src), domain_loss(
        pred_tar, domain_tar
    )
    loss = loss_s + loss_t
    return loss


class SupConLoss_m(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss_m, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        eps = 1e-6
        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).cuda()
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().cuda()
        else:
            mask = mask.float().cuda()

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + eps)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + eps)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
