#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from utils.dp_mechanism import cal_sensitivity, cal_sensitivity_MA, Laplace, Gaussian_Simple, Gaussian_MA
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import json


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdateDP(object):
    def __init__(self, args, dataset=None, idxs=None, client_idx=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.idxs_sample = np.random.choice(list(idxs), int(
            self.args.dp_sample * len(idxs)), replace=False)
        self.ldr_train = DataLoader(DatasetSplit(dataset, self.idxs_sample), batch_size=len(self.idxs_sample),
                                    shuffle=True)
        self.idxs = idxs
        self.client_idx = client_idx
        self.times = self.args.epochs * self.args.frac
        self.lr = args.lr
        self.noise_scale = self.calculate_noise_scale()

    def calculate_noise_scale(self):
        if self.args.dp_mechanism == 'Laplace':
            epsilon_single_query = self.args.dp_epsilon / self.times
            return Laplace(epsilon=epsilon_single_query)
        elif self.args.dp_mechanism == 'Gaussian':
            epsilon_single_query = self.args.dp_epsilon / self.times
            delta_single_query = self.args.dp_delta / self.times
            return Gaussian_Simple(epsilon=epsilon_single_query, delta=delta_single_query)
        elif self.args.dp_mechanism == 'MA':
            return Gaussian_MA(epsilon=self.args.dp_epsilon, delta=self.args.dp_delta, q=self.args.dp_sample, epoch=self.times)

    def train(self, net, malicious_id):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.args.lr_decay)
        loss_client = 0
        initial_weights = {name: param.clone()
                           for name, param in net.named_parameters()}

        # malicious client
        #if self.client_idx < 30 or self.client_idx >= 70:
        #if self.client_idx < 0:
        if self.client_idx in malicious_id:
            if self.args.attack == 'random':
                local_update = {name: param.clone()
                                for name, param in net.named_parameters()}
                local_update = self.set_weights_random(local_update)
                local_update = self.clip_weights_norm(
                    local_update, self.args.dp_clip)
                
                if self.args.dp_mechanism != 'no_dp':
                    local_update = self.top_k_add_noise(
                        local_update, self.args.top_k, self.args.threshold_factor)
                    pass
                self.lr = scheduler.get_last_lr()[0]
                return local_update, loss_client
            elif self.args.attack == 'negative':
                for images, labels in self.ldr_train:
                    images, labels = images.to(
                        self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    local_update = {name: self.lr * param.grad.clone() for name, param in net.named_parameters()}
                    optimizer.step()
                    scheduler.step()
                    loss_client = loss.item()
                
                if self.args.dp_mechanism != 'no_dp':
                    #local_update = self.clip_weights_norm(
                    #    local_update, self.args.dp_clip)
                    #local_update = self.top_k_add_noise(
                    #    local_update, self.args.top_k, self.args.threshold_factor)
                    pass
                self.lr = scheduler.get_last_lr()[0]
                return local_update, loss_client
            elif self.args.attack == 'flip':
                for images, labels in self.ldr_train:
                    modified_labels = labels.clone()
                    modified_labels = 9 - modified_labels
                    images, modified_labels = images.to(
                        self.args.device), modified_labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, modified_labels)
                    loss.backward()
                    local_update = {name: -self.lr * param.grad.clone() for name, param in net.named_parameters()}
                    optimizer.step()
                    scheduler.step()
                    loss_client = loss.item()
                if self.args.dp_mechanism != 'no_dp':
                    #local_update = self.clip_weights_norm(
                    #    local_update, self.args.dp_clip)
                    #local_update = self.top_k_add_noise(
                    #    local_update, self.args.top_k, self.args.threshold_factor)
                    pass
                self.lr = scheduler.get_last_lr()[0]
                return local_update, loss_client
            else:
                pass

        for images, labels in self.ldr_train:
            images, labels = images.to(
                self.args.device), labels.to(self.args.device)
            net.zero_grad()
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)

            loss.backward()
            local_update = {name: -self.lr * param.grad.clone() for name, param in net.named_parameters()}
            optimizer.step()
            scheduler.step()
            loss_client = loss.item()

        #local_update = {}
        #for name, param in net.named_parameters():
        #    local_update[name] = param - initial_weights[name]

        # add noises to parameters
        if self.args.dp_mechanism != 'no_dp':
            #self.print_weights_norm(local_update)
            local_update = self.clip_weights_norm(
                local_update, self.args.dp_clip)
            #self.print_weights_norm(local_update)
            
            #local_update = self.keep_top_k(local_update, self.args.top_k, self.args.threshold_factor)
            #local_update = self.add_noise(local_update)
            if self.args.threshold_factor:
                #local_update = self.keep_top_k(local_update, self.args.top_k, 2 * self.args.threshold_factor)
            
                local_update = self.top_k_add_noise(local_update, self.args.top_k, self.args.threshold_factor)
            else:
                local_update = self.add_noise(local_update)
            #self.print_weights_norm(local_update)

        self.lr = scheduler.get_last_lr()[0]
        # self.print_weights_norm(local_update)
        return local_update, loss_client

    def set_weights_random(self, state_dict):
        # Set weights to random values
        for name, param in state_dict.items():
            state_dict[name] = torch.randn_like(param)
        return state_dict

    def print_weights_norm(self, state_dict):
        if self.client_idx != 1:
            return
        # Calculate the norm of the weights
        total_norm = 0
        for param in state_dict.values():
            param_norm = param.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"Client {self.client_idx} Weight Norm: {total_norm}")

    def clip_weights_norm(self, state_dict, clip_value):
        # Calculate the total norm
        total_norm = 0
        for param in state_dict.values():
            param_norm = param.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Clip the weights if the total norm exceeds clip_value
        if total_norm > clip_value:
            for name, param in state_dict.items():
                state_dict[name] = param * (clip_value / total_norm)
        return state_dict

    def clip_gradients(self, net):
        if self.args.dp_mechanism == 'Laplace':
            # Laplace use 1 norm
            self.per_sample_clip(net, self.args.dp_clip, norm=1)
        elif self.args.dp_mechanism == 'Gaussian' or self.args.dp_mechanism == 'MA':
            # Gaussian use 2 norm
            self.per_sample_clip(net, self.args.dp_clip, norm=2)

    def per_sample_clip(self, net, clipping, norm):
        grad_samples = [x.grad_sample for x in net.parameters()]
        per_param_norms = [
            g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples
        ]
        per_sample_norms = torch.stack(
            per_param_norms, dim=1).norm(norm, dim=1)
        per_sample_clip_factor = (
            torch.div(clipping, (per_sample_norms + 1e-6))
        ).clamp(max=1.0)
        for grad in grad_samples:
            factor = per_sample_clip_factor.reshape(
                per_sample_clip_factor.shape + (1,) * (grad.dim() - 1))
            grad.detach().mul_(factor.to(grad.device))
        # average per sample gradient after clipping and set back gradient
        for param in net.parameters():
            param.grad = param.grad_sample.detach().mean(dim=0)

    def add_noise(self, state_dict):
        sensitivity = cal_sensitivity(
            self.lr, self.args.dp_clip, len(self.idxs_sample))
        if self.args.dp_mechanism == 'Laplace':
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.laplace(loc=0, scale=sensitivity * self.noise_scale,
                                                                    size=v.shape)).to(self.args.device)
        elif self.args.dp_mechanism == 'Gaussian':
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * self.noise_scale,
                                                                   size=v.shape)).to(self.args.device)
        elif self.args.dp_mechanism == 'MA':
            sensitivity = cal_sensitivity_MA(
                self.args.lr, self.args.dp_clip, len(self.idxs_sample))
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * self.noise_scale,
                                                                   size=v.shape)).to(self.args.device)
        return state_dict

    def keep_top_k(self, state_dict, top_k=1.0, threshold_factor=None):
        sensitivity = cal_sensitivity(
            self.lr, self.args.dp_clip, len(self.idxs_sample))
        if threshold_factor is None: 
            all_weights = []
    
            # 全部权重放在列表
            for k, v in state_dict.items():
                all_weights.extend(v.view(-1).tolist())
    
            # 权重排序
            all_weights = sorted(all_weights, key=abs, reverse=True)
    
            # 按比例
            if top_k <= 1.0:
                threshold = abs(all_weights[int(len(all_weights) * top_k) - 1])
            # 按数量
            elif top_k > 1.0:
                threshold = abs(all_weights[top_k - 1])
        else:
            threshold = threshold_factor * sensitivity * self.noise_scale

        for k, v in state_dict.items():
            mask = torch.abs(v) >= threshold
            state_dict[k] = v * mask

        return state_dict

    def top_k_add_noise(self, state_dict, top_k=1.0, threshold_factor=None):

        sensitivity = cal_sensitivity(
            self.lr, self.args.dp_clip, len(self.idxs_sample))

        if threshold_factor is None:
            # 先选取top k，只在选取权重添加噪声
            all_weights = []

            # 全部权重放在列表
            for k, v in state_dict.items():
                all_weights.extend(v.view(-1).tolist())

            # 权重排序
            all_weights = sorted(all_weights, key=abs, reverse=True)

            # 按比例
            if top_k <= 1.0:
                threshold = abs(all_weights[int(len(all_weights) * top_k) - 1])
            # 按数量
            elif top_k > 1.0:
                threshold = abs(all_weights[top_k - 1])

        else:
            threshold = threshold_factor * sensitivity * self.noise_scale
        #print(threshold)
        #print(sensitivity)
        #print(self.noise_scale)
        #exit()
        
        for k, v in state_dict.items():
            mask = torch.abs(v) >= threshold
            noise = torch.zeros_like(v)
            if self.args.dp_mechanism == 'Laplace':
                noise = torch.from_numpy(np.random.laplace(
                    loc=0, scale=sensitivity * self.noise_scale, size=v.shape)).to(self.args.device)
            elif self.args.dp_mechanism == 'Gaussian':
                noise = torch.from_numpy(np.random.normal(
                    loc=0, scale=sensitivity * self.noise_scale, size=v.shape)).to(self.args.device)
            elif self.args.dp_mechanism == 'MA':
                sensitivity = cal_sensitivity_MA(
                    self.args.lr, self.args.dp_clip, len(self.idxs_sample))
                noise = torch.from_numpy(np.random.normal(
                    loc=0, scale=sensitivity * self.noise_scale, size=v.shape)).to(self.args.device)
            # print(torch.norm(noise))

            state_dict[k] = (v + noise) * mask

        return state_dict


class LocalUpdateDPSerial(LocalUpdateDP):
    def __init__(self, args, dataset=None, idxs=None):
        super().__init__(args, dataset, idxs)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.lr, momentum=self.args.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.args.lr_decay)
        losses = 0
        for images, labels in self.ldr_train:
            net.zero_grad()
            index = int(len(images) / self.args.serial_bs)
            total_grads = [torch.zeros(size=param.shape).to(
                self.args.device) for param in net.parameters()]
            for i in range(0, index + 1):
                net.zero_grad()
                start = i * self.args.serial_bs
                end = (i+1) * self.args.serial_bs if (i+1) * \
                    self.args.serial_bs < len(images) else len(images)
                # print(end - start)
                if start == end:
                    break
                image_serial_batch, labels_serial_batch \
                    = images[start:end].to(self.args.device), labels[start:end].to(self.args.device)
                log_probs = net(image_serial_batch)
                loss = self.loss_func(log_probs, labels_serial_batch)
                loss.backward()
                if self.args.dp_mechanism != 'no_dp':
                    self.clip_gradients(net)
                grads = [param.grad.detach().clone()
                         for param in net.parameters()]
                for idx, grad in enumerate(grads):
                    total_grads[idx] += torch.mul(
                        torch.div((end - start), len(images)), grad)
                losses += loss.item() * (end - start)
            for i, param in enumerate(net.parameters()):
                param.grad = total_grads[i]
            optimizer.step()
            scheduler.step()
            # add noises to parameters
            if self.args.dp_mechanism != 'no_dp':
                self.add_noise(net)
            self.lr = scheduler.get_last_lr()[0]
        return net.state_dict(), losses / len(self.idxs_sample)
