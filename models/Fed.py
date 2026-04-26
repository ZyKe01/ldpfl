#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def FedWeightAvg(w, size):
    totalSize = sum(size)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w[0][k]*size[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size[i]
        # print(w_avg[k])
        w_avg[k] = torch.div(w_avg[k], totalSize)
    return w_avg


def FedWeightUpdate(net_glob, u, size):
    totalSize = sum(size)
    u_avg = {}
    for k in u[0].keys():
        u_avg[k] = u[0][k]*size[0]
    for k in u[0].keys():
        for i in range(1, len(u)):
            u_avg[k] += u[i][k] * size[i]
        # print(u_avg[k])
        u_avg[k] = torch.div(u_avg[k], totalSize)
    w_glob = {}
    for name, param in net_glob.named_parameters():
        w_glob[name] = u_avg[name] + param
    return w_glob
