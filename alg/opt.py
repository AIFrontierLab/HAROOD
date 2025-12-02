# coding=utf-8
import torch
import copy
from collections import OrderedDict
import operator
from numbers import Number


def get_params(alg, args, inner=False, alias=True, isteacher=False):
    if args.schuse:
        if args.schusech == "cos":
            initlr = args.lr
        else:
            initlr = 1.0
    else:
        if inner:
            initlr = args.inner_lr
        else:
            initlr = args.lr
    if isteacher:
        params = [
            {"params": alg[0].parameters(), "lr": args.lr_decay1 * initlr},
            {"params": alg[1].parameters(), "lr": args.lr_decay2 * initlr},
            {"params": alg[2].parameters(), "lr": args.lr_decay2 * initlr},
        ]
        return params
    if inner:
        params = [
            {"params": alg[0].parameters(), "lr": args.lr_decay1 * initlr},
            {"params": alg[1].parameters(), "lr": args.lr_decay2 * initlr},
        ]
    elif alias:
        params = [
            {"params": alg.featurizer.parameters(), "lr": args.lr_decay1 * initlr},
            {"params": alg.classifier.parameters(), "lr": args.lr_decay2 * initlr},
        ]
    else:
        params = [
            {"params": alg[0].parameters(), "lr": args.lr_decay1 * initlr},
            {"params": alg[1].parameters(), "lr": args.lr_decay2 * initlr},
        ]
    if ("DANN" in args.algorithm) or ("CDANN" in args.algorithm):
        params.append(
            {"params": alg.discriminator.parameters(), "lr": args.lr_decay2 * initlr}
        )
    if "CDANN" in args.algorithm:
        params.append(
            {"params": alg.class_embeddings.parameters(), "lr": args.lr_decay2 * initlr}
        )
    if "DDLearn" in args.algorithm:
        params.append(
            {"params": alg.aug_classifier.parameters(), "lr": args.lr_decay2 * initlr}
        )
    return params


def get_optimizer(alg, args, inner=False, alias=True, isteacher=False):
    params = get_params(alg, args, inner, alias, isteacher)
    if args.task.startswith("cross"):
        optimizer = torch.optim.Adam(
            params, lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, 0.9)
        )
    else:
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    return optimizer


def get_scheduler(optimizer, args):
    if args.task.startswith("cross"):
        return None
    if not args.schuse:
        return None
    if args.schusech == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.max_epoch * args.steps_per_epoch
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: args.lr * (1.0 + args.lr_gamma * float(x)) ** (-args.lr_decay),
        )
    return scheduler


class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """

    def __init__(
        self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            trust_coefficient=trust_coefficient,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1:  # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g["weight_decay"])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0,
                            (g["trust_coefficient"] * param_norm / update_norm),
                            one,
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)
                p.add_(mu, alpha=-g["lr"])


class ErmPlusPlusMovingAvg:
    def __init__(self, network):
        self.network = network
        self.network_sma = copy.deepcopy(network)
        self.network_sma.eval()
        self.sma_start_iter = 600
        self.global_iter = 0
        self.sma_count = 0

    def update_sma(self):
        self.global_iter += 1
        new_dict = {}
        if self.global_iter >= self.sma_start_iter:
            self.sma_count += 1
            for (name, param_q), (_, param_k) in zip(
                self.network.state_dict().items(), self.network_sma.state_dict().items()
            ):
                if "num_batches_tracked" not in name:
                    new_dict[name] = (
                        param_k.data.detach().clone() * self.sma_count
                        + param_q.data.detach().clone()
                    ) / (1.0 + self.sma_count)
        else:
            for (name, param_q), (_, param_k) in zip(
                self.network.state_dict().items(), self.network_sma.state_dict().items()
            ):
                if "num_batches_tracked" not in name:
                    new_dict[name] = param_q.detach().data.clone()
        self.network_sma.load_state_dict(new_dict)


class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data


def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        (
            torch.cat(tuple([t.view(-1) for t in dict_1_values]))
            - torch.cat(tuple([t.view(-1) for t in dict_2_values]))
        )
        .pow(2)
        .mean()
    )


class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)
