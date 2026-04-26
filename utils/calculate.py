#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch

def euclidean_distance(w1, w2):
    flattened_w1 = torch.cat([param.flatten() for param in w1.values()])
    flattened_w2 = torch.cat([param.flatten() for param in w2.values()])
    difference = flattened_w1 - flattened_w2
    distance = torch.norm(difference)    
    return distance
    
def param_add(state1, state2):
    #state1 + state2
    res = {}
    for name in state1.keys():
        res[name] = state1[name] + state2[name]
    return res

def param_subtract(state1, state2):
    #state1 - state2
    res = {}
    for name in state1.keys():
        res[name] = state1[name] - state2[name]
    return res

def param_norm(state):
    #l2 norm
    total_norm = 0
    for param in state.values():
        param_norm = param.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm