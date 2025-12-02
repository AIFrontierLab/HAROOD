# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from alg.opt import MovingAverage, l2_between_dicts


class Fishr(Algorithm):

    def __init__(self, args):
        super(Fishr, self).__init__(args)
        self.num_domains = args.domain_num - len(args.test_envs)
        self.args = args
        self.featurizer = get_fea(args)
        self.classifier = extend(
            common_network.feat_classifier(
                args.num_classes, self.featurizer.in_features, args.classifier
            )
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction="none"))
        self.ema_per_domain = [
            MovingAverage(ema=self.args.ema, oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])

        len_minibatches = [data[0].shape[0] for data in minibatches]

        all_z = self.featurizer(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.args.penalty_anneal_iters:
            penalty_weight = self.args.lam
            if self.update_count == self.args.penalty_anneal_iters != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {
            "loss": objective.item(),
            "nll": all_nll.item(),
            "penalty": penalty.item(),
        }

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(
            dict_grads, len_minibatches
        )
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(retain_graph=True, create_graph=True)

        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict()
        for name, weights in self.classifier.named_parameters():
            if hasattr(weights, "grad_batch"):
                dict_grads[name] = weights.grad_batch.clone().view(
                    weights.grad_batch.size(0), -1
                )
            else:
                # Fallback: compute individual gradients manually
                batch_size = logits.size(0)
                individual_grads = []
                for i in range(batch_size):
                    self.optimizer.zero_grad()
                    sample_loss = self.bce_extended(logits[i : i + 1], y[i : i + 1])
                    sample_grads = torch.autograd.grad(
                        sample_loss, weights, retain_graph=True, create_graph=True
                    )[0]
                    individual_grads.append(sample_grads.view(-1))
                dict_grads[name] = torch.stack(individual_grads)
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx : all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (
                    (env_grads_centered).pow(2).mean(dim=0)
                )

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0,
                    ).mean(dim=0),
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains

    def predict(self, x):
        return self.network(x)
