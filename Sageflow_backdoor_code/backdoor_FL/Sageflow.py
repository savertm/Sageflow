#The codes are based on Ubuntu 16.04 with Python 3.7 and Pytorch 1.0.1


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import pdb
import torch

from update import LocalUpdate, test_inference, DatasetSplit,backdoor_test_inference
from model import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, VGGCifar
from utils1 import *
from resnet import *
from options import args_parser
import csv
from torch.utils.data import DataLoader, Dataset

import os


# For experiments with both stragglers and adversaries under backdoor attack


if __name__ == '__main__':


    args = args_parser()

    start_time = time.time()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    path_project = os.path.abspath('..')

    exp_details(args)

    gpu_number = args.gpu_number
    device = torch.device(f'cuda:{gpu_number}' if args.gpu else 'cpu')

    train_dataset, test_dataset, (user_groups, dict_common) = get_dataset(args)

    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':

            if args.detail_model == 'simplecnn':
                global_model = CNNCifar(args=args)
            elif args.detail_model == 'vgg':
                global_model = VGGCifar()
            elif args.detail_model == 'resnet':
                global_model = ResNet18()

    elif args.model == 'MLP':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)

    else:
        exit('Error: unrecognized model')

    global_model.to(device)
    global_model.train()
    print(global_model)

    global_weights = global_model.state_dict()

    train_loss, train_accuracy = [], []
    final_test_acc = []
    final_backdoor_acc = []
    print_every = 1

    pre_weights = {}
    for i in range(args.staleness + 1):
        if i != 0:
            pre_weights[i] = []

    # Device schedular
    scheduler = {}

    for l in range(args.num_users):
        scheduler[l] = 0

    for epoch in tqdm(range(args.epochs)):

        local_weights_delay = {}
        for i in range(args.staleness + 1):
            local_weights_delay[i] = []

        print(f'\n | Global Training Round : {epoch + 1} | \n')

        global_model.train()


        m = max(int(args.frac * args.num_users), 1)

        # Random selection of clients
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # After a round, each staleness group is adjusted
        local_delay_ew = copy.deepcopy(pre_weights[1])

        for i in range(args.staleness + 1):
            if i != 0 and i != args.staleness:
                pre_weights[i] = copy.deepcopy(pre_weights[i + 1])

        pre_weights[args.staleness] = []


        n = int(m * args.attack_ratio)
        attack_users = []

        #The starting round of backdoor attack
        if args.dataset == 'cifar':
            if epoch >= 1000:
                attack_users = np.random.choice(idxs_users, n, replace=False)
        else:
            if epoch >= 10:
                attack_users = np.random.choice(idxs_users, n, replace=False)

        ensure_1 = 0

        global_weights_pre = copy.deepcopy(global_weights)

        for idx in idxs_users:

            if scheduler[idx] == 0:
                if idx in attack_users and args.data_poison == True:
                    local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], idx=idx,
                                              data_poison=True)
                else:

                    local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], idx=idx,
                                              data_poison=False)

                # Ensure each staleness group has at least one element
                if ensure_1 in range(args.staleness + 1):
                    delay = ensure_1
                else:
                    delay = np.random.randint(0, args.staleness + 1)

                scheduler[idx] = delay + 1


            else:

                continue

            if idx in attack_users and args.data_poison == True:

                w, loss = local_model.update_weights(

                    model=copy.deepcopy(global_model), global_round=epoch, local_ep=5, backdoor=True

                )

            else:
                w, loss = local_model.update_weights(

                    model=copy.deepcopy(global_model), global_round=epoch, local_ep=5, backdoor=False

                )

            if idx in attack_users and args.data_poison ==True:
                w_comm = communication_w(w, global_weights_pre,backdoor=True, gamma=args.backdoor_scale)
            else:
                w_comm = communication_w(w, global_weights_pre, backdoor=False)

            local_weights_delay[scheduler[idx] - 1].append(copy.deepcopy(w_comm))


            ensure_1 += 1

        for i in range(args.staleness + 1):
            if i != 0:
                if args.update_rule == 'Sageflow':
                    pre_weights[i].append({epoch: [averge_weights_comm_eflw(local_weights_delay[i], global_weights_pre, copy.deepcopy(global_model), args, train_dataset, dict_common, epoch, eta=1), len(local_weights_delay[i])]})
                else:
                    pre_weights[i].append(
                        {epoch: [average_weights_comm_fedavg(local_weights_delay[i], global_weights_pre, args.update_rule),
                                 len(local_weights_delay[i])]})


        # Staleness-aware grouping
        if args.update_rule == 'Sageflow':
            global_weights, comm = Sag(epoch, averge_weights_comm_eflw(local_weights_delay[0], global_weights_pre, copy.deepcopy(global_model), args, train_dataset, dict_common, epoch, eta=1), len(local_weights_delay[0]) ,local_delay_ew,copy.deepcopy(global_weights), args)
        else:
            global_weights, comm = Sag(epoch, average_weights_comm_fedavg(local_weights_delay[0], global_weights_pre, args.update_rule), len(local_weights_delay[0]),
                                                         local_delay_ew,copy.deepcopy(global_weights), args)



        # Update global weights
        global_model.load_state_dict(global_weights)

        list_acc, list_loss = [], []
        global_model.eval()

        for c in range(args.num_users):
            if c in attack_users and args.data_poison == True:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[c], data_poison=True, idx=c)
            else:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[c], data_poison=False, idx=c)

            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_test_acc = sum(list_acc) / len(list_acc)
        train_accuracy.append(train_test_acc)

        # if (epoch + 1) % print_every == 0:
        #     print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
        #
        #     print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

        test_acc, test_loss, _ = test_inference(args, global_model, test_dataset)
        backdoor_acc, backdoor_loss, _ = backdoor_test_inference(args, global_model, test_dataset)
        final_test_acc.append(test_acc)
        final_backdoor_acc.append(backdoor_acc)
        print('Test Accuracy: {:.2f}% \n'.format(100 * test_acc))
        print('Backdoor Test Accuracy: {:.2f}% \n'.format(100 * backdoor_acc))

        # Schedular update
        for l in range(args.num_users):
            scheduler[l] = (scheduler[l] - 1) * ((scheduler[l] - 1) > 0)

    print(f' \n Results after {args.epochs} global rounds of training:')

    print("|---- Avg testing Accuracy across each device's data: {:.2f}%".format(100 * train_accuracy[-1]))

    for i in range(len(train_accuracy)):
        print("|----{}th round Training Accuracy : {:.2f}%".format(i, 100 * train_accuracy[i]))  # Attack users

    print("|----Final Test Accuracy: {:.2f}%".format(100 * test_acc))


    for i in range(len(final_test_acc)):
        print("|----{}th round test Accuracy : {:.2f}%".format(i, 100 * final_test_acc[i]))

    for i in range(len(final_backdoor_acc)):
        print("|----{}th round backdoor success rate : {:.2f}%".format(i, 100 * final_backdoor_acc[i]))


    exp_details(args)
    file_n = f'accuracy_{args.update_rule}_{args.dataset}_backdoor_scaled_{args.backdoor_scale}_{args.seed}.csv'

    f = open(file_n, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for i in range((len(final_test_acc))):
        wr.writerow([i + 1, final_test_acc[i] * 100, final_backdoor_acc[i] * 100])

    f.close()











