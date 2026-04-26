from models.test import test_img
from models.Nets import MLP, CNNMnist, DeepCNNMnist, CNNCifar, CNNFemnist, CharLSTM
from utils.options import args_parser
import torch
from torch.utils.data import Subset, DataLoader
import numpy as np
import copy
from utils.calculate import euclidean_distance, param_add, param_subtract, param_norm

import matplotlib
matplotlib.use('Agg')

args = args_parser()
args.device = torch.device('cuda:{}'.format(
    args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
args.device = torch.device('cuda:7')

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


def cosine_similarity(weight_changes1, weight_changes2):
    """
    Calculate the cosine similarity between two dictionaries of weight changes, treating all weights as a single vector.

    Parameters:
    weight_changes1 (dict): The first dictionary of weight changes, with keys as weight names and values as weight tensors.
    weight_changes2 (dict): The second dictionary of weight changes, with keys as weight names and values as weight tensors.

    Returns:
    float: The cosine similarity between the concatenated weight vectors.
    """
    # Concatenate all flattened weight tensors into a single vector
    all_weights1 = torch.cat([weight.view(-1)
                             for weight in weight_changes1.values()])
    all_weights2 = torch.cat([weight.view(-1)
                             for weight in weight_changes2.values()])
    all_weights1 = all_weights1.to(dtype=torch.float64)
    all_weights2 = all_weights2.to(dtype=torch.float64)

    # Compute dot product
    dot_product = torch.dot(all_weights1, all_weights2).item()

    # Compute norms
    norm1 = torch.norm(all_weights1).item()
    norm2 = torch.norm(all_weights2).item()

    # Compute cosine similarity
    if norm1 > 0 and norm2 > 0:  # Prevent division by zero
        similarity = dot_product / (norm1 * norm2)
    else:
        similarity = 0.0

    return similarity


def calculating_similarity(net_glob, u_locals, dataset_server):
    model_copy = copy.deepcopy(net_glob)
    model_copy = model_copy.to(args.device)
    model_copy.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_copy = torch.optim.SGD(model_copy.parameters(), lr=args.lr)
    dataloader_server = DataLoader(
        dataset_server, batch_size=len(dataset_server), shuffle=False)
    initial_weights = {name: param.clone()
                       for name, param in net_glob.named_parameters()}
    for images, labels in dataloader_server:
        images, labels = images.to(
            args.device), labels.to(args.device)
        optimizer_copy.zero_grad()
        outputs = model_copy(images)
        loss = criterion(outputs, labels)
        loss.backward()
        weight_changes = {name: -args.lr * param.grad.clone() for name, param in model_copy.named_parameters()}
        optimizer_copy.step()
        
    total_norm = 0
    for param in weight_changes.values():
        param_norm = param.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    #weight_changes = {}
    #for name in model_copy.state_dict().keys():
    #    weight_changes[name] = model_copy.state_dict()[name] - initial_weights[name]
    '''for k, v in weight_changes.items():
        noise = torch.zeros_like(v)
        noise = torch.from_numpy(np.random.normal(loc=0, scale=0.0094818631272425, size=v.shape)).to(args.device)
        weight_changes[k] = (v + noise)'''

    result = []
    for u_local in u_locals:
        assert model_copy.state_dict().keys() == u_local.keys(
        ), "Local and global models are structured differently"
        similarity = cosine_similarity(u_local, weight_changes)
        #print(similarity)
        result.append(similarity)
    return result, total_norm


def loss_diff(net_glob, u_locals, dataset_server):
    model_copy = copy.deepcopy(net_glob)
    model_copy = model_copy.to(args.device)
    model_copy.eval()
    acc_t, loss_t = test_img(model_copy, dataset_test, args)

    result = []
    for u_local in u_locals:
        net_local_state = {name: param.clone()
                           for name, param in net_glob.named_parameters()}
        for name in u_local.keys():
            net_local_state[name] += u_local[name]
        net_local = CNNMnist(args=args).to(args.device)
        net_local.load_state_dict(net_local_state)
        net_local.eval()
        acc_l, loss_l = test_img(net_local, dataset_test, args)
        result.append(1/(loss_t-loss_l))
        # print(f"acc: {acc_l-acc_t}, loss: {loss_t-loss_l}")
    return result


def normalize_list(lst, n=1):
    # normalize to n
    lst = [max(x, 0) for x in lst]
    #lst = [1 / (1 + np.exp(-x)) for x in lst]
    # lst = [np.exp(1000*x) for x in lst]
    total = sum(lst)
    if total == 0:
        print("The total sum of the list elements is zero. Cannot normalize.")
        lst1 = [n/len(lst) for _ in range(len(lst))]
        return lst1
    normalized_lst = [x * n / total for x in lst]
    return normalized_lst


def update_weight_mine(u_locals, idxs_users, n_groups, net_glob, dataset_server, weight_locals, iter, begin_group, lamb):
    idxs_users = list(range(len(u_locals)))
    # into n groups
    # np.random.shuffle(u_locals)
    '''order = np.random.permutation(len(u_locals))
    # order = list(range(len(u_locals)))
    u_locals = [u_locals[i] for i in order]
    idxs_users = [idxs_users[i] for i in order]'''
    similaritys_best = None
    idxs_users_best = None
    similarity_max = -1
    if iter < 0:
        n_groups = int(args.frac * args.num_users)

    for _ in range(5):
        # add some random
        # combine weight, update and idx
        # sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)
        combined = list(zip(weight_locals, u_locals, idxs_users))
        prob = np.array(weight_locals)
        prob = np.array([(p+1e-6)**6 for p in prob])
        probabilities = prob / prob.sum()
        sorted_combined = []
        for _ in range(len(weight_locals)):
            index = np.random.choice(len(combined), p=probabilities)
            sorted_combined.append(combined[index])
            combined.pop(index)
            probabilities = np.delete(probabilities, index)
            probabilities = probabilities / probabilities.sum()
        _, u_locals_, idxs_users_ = zip(*sorted_combined)              
        #u_locals_, idxs_users_ = u_locals, idxs_users        

        group_size = len(u_locals_) // n_groups
        remainder = len(u_locals_) % n_groups

        groups = []
        start_index = 0
        for i in range(n_groups):
            end_index = start_index + group_size + (1 if i < remainder else 0)
            groups.append(u_locals_[start_index:end_index])
            start_index = end_index

        # Calculate the sum of updates for each group
        update_groups = []
        for i in range(n_groups):
            update_group = {}
            total_weight = 0
            for name in groups[i][0].keys():
                update_group[name] = 0
            for j in range(len(groups[i])):
                u_local = groups[i][j]
                idx = idxs_users_[i * group_size + j]
                total_weight += weight_locals[idx]
                for name in u_local.keys():
                    update_group[name] += u_local[name] * weight_locals[idx]
            for name in update_group.keys():
                update_group[name] = torch.div(
                    update_group[name], total_weight)
            update_groups.append(update_group)

        # similaritys is a list, len(similaritys) == n_groups
        similaritys, _ = calculating_similarity(net_glob, update_groups, dataset_server)

        if max(similaritys) > similarity_max:
            similarity_max = max(similaritys)
            similaritys_best = similaritys
            idxs_users_best = idxs_users_
    #print(similaritys_best)
    similaritys_best = normalize_list(similaritys_best, sum(weight_locals) / group_size)
    
    for i in range(len(similaritys_best)):
        for j in range(group_size):
            idx = idxs_users_best[i * group_size + j]
            weight_locals[idx] = max(lamb * weight_locals[idx] + \
                (1 - lamb) * similaritys_best[i], 0)
    
    if iter >= 50:
        total_weight = sum(weight_locals)
        weight_locals = [0 if x < 0.010 else x for x in weight_locals]
        weight_locals = normalize_list(weight_locals, n=total_weight)
    elif iter >= 0:
        total_weight = sum(weight_locals)
        weight_locals = [0 if x < 0.010 else x for x in weight_locals]
        weight_locals = normalize_list(weight_locals, n=total_weight)
    return weight_locals

def update_weight_RoWA(u_locals, idxs_users, n_groups, net_glob, dataset_server, weight_locals, iter):
    # RoWA
    g_locals = []
    for u_local in u_locals:
        g_local = {name: -param / args.lr for name, param in u_local.items()}
        all_gradients = torch.cat([param.flatten() for param in g_local.values()])
        total_norm = torch.norm(all_gradients).item()
        total_norm_tensor = torch.tensor(total_norm)
        normalization_factor = torch.log1p(total_norm_tensor) / total_norm_tensor.item() if total_norm > 0 else 0
        normalized_g = {name: normalization_factor * param for name, param in g_local.items()}

        g_locals.append(normalized_g)
    
    current_weights = {name: param.data.clone() for name, param in net_glob.named_parameters()}
    updated_params = []
    for g_local in g_locals:
        updated_param = {}
        with torch.no_grad():
            for name, param in net_glob.named_parameters():
                updated_param[name] = current_weights[name] - args.lr * g_local[name]
        updated_params.append(updated_param)

    robust_scores = []
    for updated_param in updated_params:
        score = cosine_similarity(current_weights, updated_param)
        score = max(score, 0)
        robust_scores.append(score)

    return robust_scores
      
def update_weight_FLtrust(u_locals, idxs_users, n_groups, net_glob, dataset_server, weight_locals, iter):
    # similaritys is a list, len(similaritys) == n_groups
    similaritys, total_norm = calculating_similarity(
        net_glob, u_locals, dataset_server)

    similaritys = normalize_list(similaritys, 1)
    print(similaritys)
    lamb = 0
    for i in range(len(similaritys)):
        weight_locals[i] = lamb * weight_locals[i] + (1 - lamb) * similaritys[i]

    return weight_locals, total_norm

        
def update_weight_Krum(u_locals, n):
    distances_sum_and_indices = []
    for i in range(len(u_locals)):
        u_local = u_locals[i]
        distances = []
        for j in range(len(u_locals)):
            if i != j:
                dist = euclidean_distance(u_local, u_locals[j])
                distances.append((dist.item(), j))

        distances.sort()
        nearest_n_distances_sum = sum(dist for dist, _ in distances[:n])
        nearest_n_indices = [i] + [idx for _, idx in distances[:n]]
        distances_sum_and_indices.append((nearest_n_distances_sum, nearest_n_indices))

    distances_sum_and_indices.sort()
    indices = distances_sum_and_indices[0][1]

    weight_users = [0] * len(u_locals)
    for idx in indices:
        weight_users[idx] = 1 / (n + 1)

    return weight_users

def update_weight_tdsc(u_locals, idxs_users, n_groups, net_glob, dataset_server, weight_locals, iter):
    # norm detection
    beta1 = 1.0
    current_weights = {name: param.data.clone() for name, param in net_glob.named_parameters()}
    param_locals = []
    for u_local in u_locals:
        param_local = {}
        with torch.no_grad():
            for name in current_weights.keys():
                param_local[name] = current_weights[name] + u_local[name]
        param_locals.append(param_local)

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

        
    #accuracy detection
    beta2 = 0.05
    num_classes = 10
    rate_locals = []
    for i in range(N):
        d_norm = param_norm(param_subtract(param_locals[i], param_stds[i]))
        d = d_norm ** 2
        error1 = d/(param_norm(param_locals[i]) ** 2)
        rate_norm = 1 - max(0, error1 - beta1)
        #print(error1)
        
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
            error2 = max(error2, delta[k])
        #print(accuracy_std)
        #print(accuracy_local)
        if error2 > beta2:
            rate_acc = 1 - error2
        #print(error2)
            
        # mix
        gamma = 0.5
        rate_mix = gamma * rate_norm + (1 - gamma) * rate_acc
        rate_locals.append(rate_mix)
    #print(rate_locals)
    rate_locals_tensor = torch.tensor(rate_locals)
    weight_users = torch.nn.functional.softmax(rate_locals_tensor, dim=0).numpy().tolist()
    return weight_users