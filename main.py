#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from opacus.grad_sample import GradSampleModule
from utils.dataset import FEMNIST, ShakeSpeare
from models.test import test_img
from models.Fed import FedAvg, FedWeightAvg, FedWeightUpdate
from models.Nets import MLP, CNNMnist, DeepCNNMnist, CNNCifar, CNNFemnist, CharLSTM
from models.Update import LocalUpdateDP, LocalUpdateDPSerial
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
from models.Defense import split_server_client, cosine_similarity, calculating_similarity, normalize_list, update_weight_mine, update_weight_FLtrust, update_weight_RoWA, update_weight_Krum, update_weight_tdsc

import matplotlib
matplotlib.use('Agg')

        
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
    args.device = torch.device('cuda:7')
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
        clients = [LocalUpdateDPSerial(
            args=args, dataset=dataset_train, idxs=dict_users[i]) for i in range(args.num_users)]
    else:
        clients = [LocalUpdateDP(args=args, dataset=dataset_train,
                                 idxs=dict_users[i], client_idx=i) for i in range(args.num_users)]
    m, loop_index = max(int(args.frac * args.num_users), 1), int(1 / args.frac)

    weight_locals = [1/(args.frac * args.num_users) for i in range(args.num_users)]
    #weight_locals=[0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025, 0, 0, 0, 0, 0, 0, 0.025, 0.025, 0.025, 0.025]
    
    #malicious_id = random.sample(range(100), 60)
    #malicious_id=[0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34, 35, 40, 41, 42, 43, 44, 45, 50, 51, 52, 53, 54, 55, 60, 61, 62, 63, 64, 65, 70, 71, 72, 73, 74, 75, 80, 81, 82, 83, 84, 85, 90, 91, 92, 93, 94, 95]
    malicious_id=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89]
    malicious_id=[0, 1, 2, 3, 4, 5, 6, 7, 25, 26, 27, 28, 29, 30, 31, 50, 51, 52, 53, 54, 55, 56, 57, 75, 76, 77, 78, 79, 80, 81]
    #malicious_id = [0, 1, 2, 3, 4, 25, 26, 27, 28, 29, 50, 51, 52, 53, 54, 75, 76, 77, 78, 79]
    
    for iter in range(args.epochs):
        t_start = time.time()
        u_locals, loss_locals = [], []
        # round-robin selection
        begin_index = (iter % loop_index) * m
        end_index = begin_index + m
        idxs_users = all_clients[begin_index:end_index]
        weight_users = weight_locals[begin_index:end_index]
        for idx in idxs_users:
            local = clients[idx]
            u, loss = local.train(net=copy.deepcopy(net_glob).to(
                args.device), malicious_id=malicious_id)
            u_locals.append(u)
            loss_locals.append(copy.deepcopy(loss))

        # update global weights
        # calculating_similarity(copy.deepcopy(net_glob), u_locals, dataset_server)
        # w_glob = FedWeightAvg(w_locals, weight_locols)
        if args.robust == 'mine':
            begin_group = args.begin_group
            lamb = 0.75#args.lamb
            weight_users = update_weight_mine(
                u_locals, idxs_users, 5, net_glob, dataset_server, weight_users, iter, begin_group, lamb)
        elif args.robust == 'RoWA':
            weight_users = update_weight_RoWA(
                u_locals, idxs_users, 0, net_glob, dataset_server, weight_users, iter)
        elif args.robust == 'FLtrust':
            weight_users, server_norm = update_weight_FLtrust(
                u_locals, idxs_users, 0, net_glob, dataset_server, weight_users, iter)
            scaled_u_locals = []
            for u_local in u_locals:
                total_norm = 0
                for param in u_local.values():
                    param_norm = param.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                if total_norm > 0:
                    #scaling_factor = server_norm / total_norm
                    scaling_factor = 1
                else:
                    scaling_factor = 1
                for name, param in u_local.items():
                    u_local[name] = param * scaling_factor
                scaled_u_locals.append(u_local)
            #u_locals = scaled_u_locals
            
        elif args.robust == 'Krum':
            n = int(args.frac*(args.num_users-len(malicious_id))-1)
            weight_users = update_weight_Krum(u_locals, n)
            
        elif args.robust == 'tdsc':
            weight_users = update_weight_tdsc(u_locals, idxs_users, 5, net_glob, dataset_server, weight_locals, iter)
                
        # weight selected
        weight_locals[begin_index:end_index] = weight_users
        weight_users = weight_locals[begin_index:end_index]
        w_glob = FedWeightUpdate(net_glob, u_locals, weight_users)
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
    
    if args.robust == 'mine':
        accfile = open(rootpath + '/accfile_fed_{}_{}_{}_iid{}_dp_{}_epsilon_{}_threshold_{}_robust_{}_{}_lambda_{}_before{}.dat'.
                   format(args.dataset, args.model, args.epochs, args.iid,
                          args.dp_mechanism, args.dp_epsilon, args.threshold_factor, args.robust, args.attack, args.lamb, args.begin_group), "w")
    else:
        accfile = open(rootpath + '/accfile_fed_{}_{}_{}_iid{}_dp_{}_epsilon_{}_threshold_{}_robust_{}_{}.dat'.
                   format(args.dataset, args.model, args.epochs, args.iid,
                          args.dp_mechanism, args.dp_epsilon, args.threshold_factor, args.robust, args.attack), "w")

    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()

    # plot loss curve
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel('test accuracy')
    plt.savefig(rootpath + '/fed_{}_{}_{}_C{}_iid{}_dp_{}_epsilon_{}_threshold_{}_robust_{}_{}_acc.png'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid, args.dp_mechanism, args.dp_epsilon, args.threshold_factor, args.robust, args.attack))
