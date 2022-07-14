#The codes are based on Ubuntu 16.04 with Python 3.7 and Pytorch 1.0.1

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch

from update import LocalUpdate, test_inference, DatasetSplit
from model import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, VGGCifar
from resnet import *
from utils1 import *
import csv
from torch.utils.data import DataLoader, Dataset
from options import args_parser
import os


# For experiments with only stragglers
# For experiments with both stragglers and adversaries


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

    train_accuracy = []
    final_test_acc = []
    print_every = 1

    pre_weights = {}
    for i in range(args.staleness + 1):
        if i != 0:
            pre_weights[i] = []

    # Device schedular
    scheduler = {}

    for l in range(args.num_users):
        scheduler[l] = 0

    global_epoch = 0

    for epoch in tqdm(range(args.epochs)):

        local_weights_delay = {}
        loss_on_public = {}
        entropy_on_public = {}

        for i in range(args.staleness + 1):
            loss_on_public[i] = []
            entropy_on_public[i] = []
            local_weights_delay[i] = []


        print(f'\n | Global Training Round : {epoch + 1} | \n')

        global_model.train()

        m = max(int(args.frac * args.num_users), 1)

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        n = int(m * args.attack_ratio)
        attack_users = np.random.choice(idxs_users, n, replace=False)




        global_weights_rep = copy.deepcopy(global_model.state_dict())

        # After round, each staleness group is adjusted
        local_delay_ew = copy.deepcopy(pre_weights[1])

        for i in range(args.staleness + 1):
            if i != 0 and i != args.staleness:
                pre_weights[i] = copy.deepcopy(pre_weights[i+1])

        pre_weights[args.staleness] = []


        ensure_1 = 0

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

            w, loss = local_model.update_weights(

                model=copy.deepcopy(global_model), global_round=epoch

            )

            if idx in attack_users and args.model_poison == True:
                w = sign_attack(w, args.model_poison_scale)

            ensure_1 += 1

            global_model.load_state_dict(w)

            # Compute the loss and entropy for each device on public dataset
            common_acc, common_loss_sync, common_entropy_sample = test_inference(args, global_model,
                                                                                 DatasetSplit(train_dataset,
                                                                                       dict_common))
            local_weights_delay[ scheduler[idx] - 1 ].append(copy.deepcopy(w))
            loss_on_public[scheduler[idx] - 1].append(common_loss_sync)
            entropy_on_public[scheduler[idx] - 1].append(common_entropy_sample)

            global_model.load_state_dict(global_weights_rep)

        for i in range(args.staleness + 1):
            if i != 0:
                if args.update_rule == 'Sageflow':
                    # Averaging delayed local weights via entropy-based filtering and loss-wegithed averaging
                    w_avg_delay, len_delay = Eflow(local_weights_delay[i], loss_on_public[i], entropy_on_public[i], epoch)
                    pre_weights[i].append({epoch: [w_avg_delay, len_delay]})

                else:
                    pre_weights[i].append({epoch: [average_weights(local_weights_delay[i]), len(local_weights_delay[i])]})

        if args.update_rule == 'Sageflow':
            # Averaging current local weights via entropy-based filtering and loss-wegithed averaging
            sync_weights, len_sync = Eflow(local_weights_delay[0], loss_on_public[0], entropy_on_public[0], epoch)

            # Staleness-aware grouping
            global_weights = Sag(epoch, sync_weights, len_sync, local_delay_ew,
                                                     copy.deepcopy(global_weights))

        else:
            global_weights = Sag(epoch, average_weights(local_weights_delay[0]), len(local_weights_delay[0]),
                                                 local_delay_ew, copy.deepcopy(global_weights))

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

        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')

            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

        test_acc, test_loss, _ = test_inference(args, global_model, test_dataset)
        final_test_acc.append(test_acc)
        print('Test Accuracy: {:.2f}% \n'.format(100 * test_acc))

        # Schedular Update
        for l in range(args.num_users):
            scheduler[l] = (scheduler[l] - 1) * ((scheduler[l] - 1) > 0)


    print(f' \n Results after {args.epochs} global rounds of training:')

    print("|---- Avg testing Accuracy across each device's data: {:.2f}%".format(100 * train_accuracy[-1]))

    for i in range(len(train_accuracy)):
        print("|----{}th round Training Accuracy : {:.2f}%".format(i, 100 * train_accuracy[i]))

    print("|----Final Test Accuracy: {:.2f}%".format(100 * test_acc))

    for i in range(len(final_test_acc)):
        print("|----{}th round Final Test Accuracy : {:.2f}%".format(i, 100 * final_test_acc[i]))


    exp_details(args)
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))


    if args.data_poison == True:
        attack_type = 'data'
    elif args.model_poison == True:
        attack_type = 'model'
        model_scale = '_scale_' + str(args.model_poison_scale)
        attack_type += model_scale
    else:
        attack_type = 'no_attack'

    file_n = f'accuracy_{args.update_rule}__{args.dataset}_{attack_type}_poison_eth_{args.eth}_delta_{args.delta}_{args.frac}_{args.seed}_{args.lam}.csv'

    f = open(file_n, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for i in range((len(final_test_acc))):
        wr.writerow([i + 1, final_test_acc[i] * 100])

    f.close()










