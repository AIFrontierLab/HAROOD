# coding=utf-8
import torch
from network import act_network
import numpy as np
from sklearn.metrics import f1_score


def get_fea(args):
    if args.model_size == "transformer":
        if args.algorithm == "LAG":
            net = act_network.transformfea_LAG(args)
        else:
            net = act_network.transformfea(args)
        return net
    if args.task == "cross_people" or args.task == "cross_time":
        if args.model_size == "small":
            net = act_network.SActNetwork(args.dataset)
        elif args.model_size == "large":
            net = act_network.LActNetwork(args.dataset)
        elif args.model_size == "rnn":
            net = act_network.SimpleTwoLayerRNN(args.dataset, rnn_type="RNN")
        elif args.model_size == "lstm":
            net = act_network.SimpleTwoLayerRNN(args.dataset, rnn_type="LSTM")
        else:
            if args.algorithm == "LAG":
                net = act_network.ActNetwork_LAG_CNN(args.dataset, args.grid_size)
            else:
                net = act_network.ActNetwork(args.dataset)
    elif args.task == "cross_position":
        if args.algorithm == "LAG":
            net = act_network.ActNetwork_LAG_CNN("p" + args.dataset, args.grid_size)
        else:
            net = act_network.ActNetwork("p" + args.dataset)
    elif args.task == "cross_device":
        if args.algorithm == "LAG":
            net = act_network.ActNetwork_LAG_CNN("d" + args.dataset, args.grid_size)
        else:
            net = act_network.ActNetwork("d" + args.dataset)
    else:
        if args.algorithm == "LAG":
            net = act_network.ActNetwork_LAG_CNN(args.task, args.grid_size)
        else:
            net = act_network.ActNetwork(args.task)
    return net


def accuracy(network, loader):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            p = network.predict(x)

            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
    network.train()
    return correct / total


def acc_f1(network, loader):
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            p = network.predict(x)

            if p.size(1) == 1:
                preds = p.gt(0).float()
                correct += (preds.eq(y).float()).sum().item()
            else:
                preds = p.argmax(1)
                correct += (preds.eq(y).float()).sum().item()

            # Collect predictions and targets for F1 calculation
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            total += len(x)

    # Calculate accuracy
    accuracy = correct / total

    # Calculate F1 score using sklearn
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    # For binary classification, use binary F1
    # For multi-class, use macro F1 by default
    if p.size(1) == 1:
        f1 = f1_score(all_targets, all_predictions, average="binary")
    else:
        f1 = f1_score(all_targets, all_predictions, average="macro")

    network.train()
    return accuracy, f1
