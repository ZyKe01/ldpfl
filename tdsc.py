#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from opacus.grad_sample import GradSampleModule
from utils.dataset import FEMNIST, ShakeSpeare
from models.test import test_img
from models.Fed import FedAvg, FedWeightAvg, FedWeightUpdate
from models.Nets import MLP, CNNMnist, DeepCNNMnist, CNNCifar, CNNFemnist, CharLSTM
from utils.options import args_parser
import os
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import copy
import matplotlib.pyplot as plt
import random
import time
from utils.calculate import euclidean_distance, param_add, param_subtract, param_norm
from torch import nn, autograd
from utils.dp_mechanism import cal_sensitivity, cal_sensitivity_MA, Laplace, Gaussian_Simple, Gaussian_MA
from torch.utils.data import DataLoader, Dataset
import numpy as np

import matplotlib
matplotlib.use('Agg')

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

        # malicious client
        if self.client_idx < 30 or self.client_idx >= 70:
        #if self.client_idx < 0:
        #if self.client_idx in malicious_id:
            if self.args.attack == 'random':
                net_local = {name: param.clone()
                                for name, param in net.named_parameters()}
                net_local = self.set_weights_random(net_local)
                net_local = self.clip_weights_norm(
                    net_local, self.args.dp_clip)
                
                if self.args.dp_mechanism != 'no_dp':
                    net_local = self.add_noise(net_local)
                    pass
                self.lr = scheduler.get_last_lr()[0]
                return net_local, loss_client
            elif self.args.attack == 'negative':
                for images, labels in self.ldr_train:
                    images, labels = images.to(
                        self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    loss_client = loss.item()
                
                net_local = net.state_dict()
                if self.args.dp_mechanism != 'no_dp':
                    #net_local = self.clip_weights_norm(
                    #    net_local, self.args.dp_clip)
                    #net_local = self.add_noise(net_local)
                    pass
                self.lr = scheduler.get_last_lr()[0]
                return net_local, loss_client
            elif self.args.attack == 'flip':
                #print('flip')
                for images, labels in self.ldr_train:
                    modified_labels = labels.clone()
                    modified_labels = 9 - modified_labels
                    images, modified_labels = images.to(
                        self.args.device), modified_labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, modified_labels)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    loss_client = loss.item()

                net_local = net.state_dict()
                if self.args.dp_mechanism != 'no_dp':
                    #net_local = self.clip_weights_norm(
                    #    net_local, self.args.dp_clip)
                    #net_local = self.add_noise(net_local)
                    pass
                self.lr = scheduler.get_last_lr()[0]
                return net_local, loss_client
            else:
                pass

        for images, labels in self.ldr_train:
            images, labels = images.to(
                self.args.device), labels.to(self.args.device)
            net.zero_grad()
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)

            loss.backward()
            #local_update = {name: -self.lr * param.grad.clone() for name, param in net.named_parameters()}
            optimizer.step()
            scheduler.step()
            loss_client = loss.item()

        net_local = net.state_dict()
        # add noises to parameters
        if self.args.dp_mechanism != 'no_dp':
            #self.print_weights_norm(local_update)
            self.print_weights_norm(net_local)
            net_local = self.clip_weights_norm(
                net_local, self.args.dp_clip)
            net_local = self.add_noise(net_local)
            self.print_weights_norm(net_local)

        self.lr = scheduler.get_last_lr()[0]
        # self.print_weights_norm(net_local)
        return net_local, loss_client

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

def split_server_client(dataset, num_server):
    # Separate out the validation dataset used by the server
    np.random.seed(42)
    num_samples = num_server
    # Generate a random index of the sample
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    server_dataset = Subset(dataset, indices)
    indices_set = set(indices)
    # Creating a new dataset (excluding specified indexes)
    filtered_indices = [i for i in range(
        len(dataset)) if i not in indices_set]
    client_dataset = Subset(dataset, filtered_indices)
    return server_dataset, client_dataset

def normalize_list(lst, n=1):
    # normalize to n
    lst = [max(x, 0) for x in lst]
    #lst = [1 / (1 + np.exp(-x)) for x in lst]
    # lst = [np.exp(1000*x) for x in lst]
    total = sum(lst)
    if total == 0:
        print("The total sum of the list elements is zero. Cannot normalize.")
        return lst
    normalized_lst = [x * n / total for x in lst]
    return normalized_lst


def update_weight_tdsc(net_locals, idxs_users, n_groups, net_glob, dataset_server, weight_locals, iter):
    # norm detection
    beta1 = 1
    param_locals = net_locals
    param_avg = {}
    for name in param_locals[0].keys(): 
        param_avg[name] = 0
    for param_local in param_locals:
        for name in param_local.keys():
            param_avg[name] += param_local[name]
    for name in param_avg.keys():
        param_avg[name] = torch.div(param_avg[name], len(param_locals))

    param_stds = []
    N = len(param_locals)
    for param_local in param_locals:
        param_std = {}
        for name in param_local.keys():
            param_std[name] = (param_avg[name] - param_local[name]/N) * (N/(N-1))
        param_stds.append(param_std)

    '''for i in range(N):
        d_norm = param_norm(param_subtract(param_locals[i], param_stds[i]))
        d = d_norm ** 2
        error1 = d/(param_norm(param_locals[i]) ** 2)
        rate_norm = 1 - max(0, error1 - beta1)
        print(rate_norm)'''
    
    #accuracy detection
    beta2 = 0.05
    num_classes = 10
    rate_locals = []
    for i in range(N):

        d_norm = param_norm(param_subtract(param_locals[i], param_stds[i]))
        d = d_norm ** 2
        error1 = d/(param_norm(param_locals[i]) ** 2)
        rate_norm = 1 - max(0, error1 - beta1)
        print(' error1 ',rate_norm)

        model_class = net_glob.__class__
        net_std = model_class(args=args).to(args.device)
        net_std.load_state_dict(param_stds[i])
        dataloader_server = DataLoader(dataset_server, batch_size=len(dataset_server), shuffle=False)
        correct_predictions_std = {i: 0 for i in range(num_classes)}
        total_predictions_std = {i: 0 for i in range(num_classes)}
        accuracy_std = []
        with torch.no_grad():
            net_std.eval()
            for images, labels in dataloader_server:
                images, labels = images.to(
                    args.device), labels.to(args.device)
                outputs = net_std(images)
                _, predicted = torch.max(outputs, 1)
                for label, pred in zip(labels, predicted):
                    label = label.item()
                    pred = pred.item()
                    if label == pred:
                        correct_predictions_std[label] += 1
                    total_predictions_std[label] += 1
        for class_id in range(num_classes):
            accuracy = correct_predictions_std[class_id] / total_predictions_std[class_id] if total_predictions_std[class_id] > 0 else 0
            accuracy_std.append(accuracy)

        net_local = model_class(args=args).to(args.device)
        net_local.load_state_dict(param_locals[i])
        correct_predictions_local = {i: 0 for i in range(num_classes)}
        total_predictions_local = {i: 0 for i in range(num_classes)}
        accuracy_local = []
        with torch.no_grad():
            net_local.eval()
            for images, labels in dataloader_server:
                images, labels = images.to(
                    args.device), labels.to(args.device)
                outputs = net_local(images)
                _, predicted = torch.max(outputs, 1)
                for label, pred in zip(labels, predicted):
                    label = label.item()
                    pred = pred.item()
                    if label == pred:
                        correct_predictions_local[label] += 1
                    total_predictions_local[label] += 1
        for class_id in range(num_classes):
            accuracy = correct_predictions_local[class_id] / total_predictions_local[class_id] if total_predictions_local[class_id] > 0 else 0
            accuracy_local.append(accuracy)

        delta = []
        for k in range(num_classes):
            if accuracy_std[k] <= accuracy_local[k]:
                delta_k = 0
                delta.append(delta_k)
            else:
                delta_k = (accuracy_std[k] - accuracy_local[k])/accuracy_std[k]
                delta.append(delta_k)
        error2 = 0
        rate_acc = 1
        for k in range(num_classes):
            #print(delta[k])
            error2 = max(error2, delta[k])
        #print(accuracy_std)
        #print(accuracy_local)
        if error2 > beta2:
            rate_acc = 1 - error2
        #print(' rate_norm ',rate_norm)
            
        # mix
        gamma = 0.5
        rate_mix = gamma * rate_norm + (1 - gamma) * rate_acc
        rate_locals.append(rate_mix)
    #print(rate_locals)
    rate_locals_tensor = torch.tensor(rate_locals)
    weight_users = torch.nn.functional.softmax(rate_locals_tensor, dim=0).numpy().tolist()
    return weight_users
        
if __name__ == '__main__':
    # parse args

    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed(seed)

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dict_users = {}
    dataset_train, dataset_test = None, None

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(
            './data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(
            './data/mnist/', train=False, download=True, transform=trans_mnist)
        args.num_channels = 1
        dataset_server, dataset_train = split_server_client(
            dataset_train, 1000)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        # trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        args.num_channels = 3
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10(
            './data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10(
            './data/cifar', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion-mnist':
        args.num_channels = 1
        trans_fashion_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                             transform=trans_fashion_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'femnist':
        args.num_channels = 1
        dataset_train = FEMNIST(train=True)
        dataset_test = FEMNIST(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: femnist dataset is naturally non-iid')
        else:
            print(
                "Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
    elif args.dataset == 'shakespeare':
        dataset_train = ShakeSpeare(train=True)
        dataset_test = ShakeSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    net_glob = None
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.dataset == 'femnist' and args.model == 'cnn':
        net_glob = CNNFemnist(args=args).to(args.device)
    elif args.dataset == 'shakespeare' and args.model == 'lstm':
        net_glob = CharLSTM().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64,
                       dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    # use opacus to wrap model to clip per sample gradient
    if args.dp_mechanism != 'no_dp':
        pass  # net_glob = GradSampleModule(net_glob)
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    all_clients = list(range(args.num_users))

    # training
    acc_test = []
    if args.serial:
        #clients = [LocalUpdateDPSerial(
        #    args=args, dataset=dataset_train, idxs=dict_users[i]) for i in range(args.num_users)]
        pass
    else:
        clients = [LocalUpdateDP(args=args, dataset=dataset_train,
                                 idxs=dict_users[i], client_idx=i) for i in range(args.num_users)]
    m, loop_index = max(int(args.frac * args.num_users), 1), int(1 / args.frac)

    weight_locals = [1/(args.frac * args.num_users) for i in range(args.num_users)]
    #weight_locals=[0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025]
    
    malicious_id = random.sample(range(100), 0)
    #malicious_id=[0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34, 35, 40, 41, 42, 43, 44, 45, 50, 51, 52, 53, 54, 55, 60, 61, 62, 63, 64, 65, 70, 71, 72, 73, 74, 75, 80, 81, 82, 83, 84, 85, 90, 91, 92, 93, 94, 95]
    malicious_id=[6, 7, 8, 9, 16, 17, 18, 19, 26, 27, 28, 29, 36, 37, 38, 39, 46, 47, 48, 49, 56, 57, 58, 59, 66, 67, 68, 69, 76, 77, 78, 79, 86, 87, 88, 89, 96, 97, 98, 99]
    
    for iter in range(args.epochs):
        t_start = time.time()
        net_locals, loss_locals = [], []
        # round-robin selection
        begin_index = (iter % loop_index) * m
        end_index = begin_index + m
        idxs_users = all_clients[begin_index:end_index]
        weight_users = weight_locals[begin_index:end_index]
        for idx in idxs_users:
            local = clients[idx]
            net_local, loss = local.train(net=copy.deepcopy(net_glob).to(
                args.device), malicious_id=malicious_id)
            net_locals.append(net_local)
            loss_locals.append(copy.deepcopy(loss))

        # update global weights
        # calculating_similarity(copy.deepcopy(net_glob), u_locals, dataset_server)
        # w_glob = FedWeightAvg(w_locals, weight_locols)
        if args.robust == 'tdsc':
            weight_users = update_weight_tdsc(net_locals, idxs_users, 5, net_glob, dataset_server, weight_locals, iter)
            w_glob = FedWeightAvg(net_locals, weight_users)
                
        # weight selected
        weight_locals[begin_index:end_index] = weight_users
        weight_users = weight_locals[begin_index:end_index]
        #w_glob = FedWeightUpdate(net_glob, u_locals, weight_users)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        print(weight_users)
        
        # print accuracy
        net_glob.eval()
        acc_t, loss_t = test_img(net_glob, dataset_test, args)
        t_end = time.time()
        print("Round {:3d},Testing accuracy: {:.2f},Time:  {:.2f}s".format(
            iter, acc_t, t_end - t_start))

        acc_test.append(acc_t.item())

    print(weight_locals)
    rootpath = './test_log'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    accfile = open(rootpath + '/accfile_fed_{}_{}_{}_iid{}_dp_{}_epsilon_{}_threshold_{}_robust{}_{}.dat'.
                   format(args.dataset, args.model, args.epochs, args.iid,
                          args.dp_mechanism, args.dp_epsilon, args.threshold_factor, args.robust,                             args.attack
                          ), "w")

    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()

    # plot loss curve
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel('test accuracy')
    plt.savefig(rootpath + '/fed_{}_{}_{}_C{}_iid{}_dp_{}_epsilon_{}_threshold_{}_robust{}_{}_acc.png'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid, args.dp_mechanism, args.dp_epsilon, args.threshold_factor, args.robust, args.attack))
