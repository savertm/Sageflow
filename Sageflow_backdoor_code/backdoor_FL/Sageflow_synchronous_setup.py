#The codes are based on Ubuntu 16.04 with Python 3.7 and Pytorch 1.0.1

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch

from options import args_parser
from update import LocalUpdate, test_inference,backdoor_test_inference
from model import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar ,VGGCifar
from utils1 import *
from resnet import *

import csv
import os


# For experiments with only adversaries under backdoor attack


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
            elif args.detail_model =='vgg':
                global_model = VGGCifar()
            elif args.detail_model =='resnet':
                global_model = ResNet18()





    elif args.model =='MLP':
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
    final_test_acc=[]
    final_backdoor_acc = []

    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} | \n')

        global_model.train()


        m = max(int(args.frac * args.num_users), 1)

        idxs_users = np.random.choice(range(args.num_users), m , replace=False)

        n = int(m*args.attack_ratio)
        attack_users = []

        #The starting round of backdoor attack
        if args.dataset=='cifar':
            if epoch >=1000:
                attack_users = np.random.choice(idxs_users, n, replace=False)
        else:
            if epoch >=10:
                attack_users = np.random.choice(idxs_users, n, replace=False)

        global_weights_pre = copy.deepcopy(global_weights)


        for idx in idxs_users:

            if idx in attack_users and args.data_poison==True:
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], idx=idx, data_poison=True)
            else:
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], idx=idx, data_poison=False)

            if idx in attack_users and args.data_poison == True:

                w, loss = local_model.update_weights(

                    model=copy.deepcopy(global_model), global_round=epoch, local_ep=10, backdoor=True

                )

            else:
                w, loss = local_model.update_weights(

                    model=copy.deepcopy(global_model), global_round=epoch , local_ep=1, backdoor=False

                )

            if idx in attack_users and args.model_poison ==True:
                w = sign_attack(w)

            if idx in attack_users and args.data_poison ==True:
                w_comm = communication_w(w, global_weights_pre,backdoor=True, gamma=args.backdoor_scale)

            else:

                w_comm = communication_w(w, global_weights_pre, backdoor=False)

            local_weights.append(copy.deepcopy(w_comm))
            local_losses.append(copy.deepcopy(loss))

        if args.update_rule == 'Sageflow':
            global_weights = averge_weights_comm_eflw(local_weights, global_weights_pre, copy.deepcopy(global_model), args, train_dataset, dict_common, epoch, eta=1)
        else:
            global_weights = average_weights_comm_fedavg(local_weights, global_weights_pre, args.update_rule)

        #Update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)

        train_loss.append(loss_avg)

        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            if c in attack_users and args.data_poison==True:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], data_poison=True , idx=c)
            else:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[c], data_poison=False, idx=c)

            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_test_acc = sum(list_acc)/len(list_acc)
        train_accuracy.append(train_test_acc)

        # if (epoch+1) % 1 == 0:
        #     print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        #     print(f'Training Loss : {np.mean(np.array(train_loss))}')
        #     print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        test_acc, test_loss,_ = test_inference(args, global_model, test_dataset)
        backdoor_acc, backdoor_loss,_ = backdoor_test_inference(args, global_model, test_dataset)

        print('Test Accuracy: {:.2f}% \n'.format(100 * test_acc))
        print('Backdoor Success Rate: {:.2f}% \n'.format(100 * backdoor_acc))
        final_test_acc.append(test_acc)
        final_backdoor_acc.append(backdoor_acc)



    print(f' \n Results after {args.epochs} global rounds of training:')

    print("|---- Avg testing Accuracy across each device's data: {:.2f}%".format(100*train_accuracy[-1]))

    for i in range(len(train_accuracy)):

        print("|----{}th round Training Accuracy : {:.2f}%".format(i,100*train_accuracy[i]))

    print("|----Final testing Accuracy: {:.2f}%".format(100*test_acc))


    for i in range(len(final_test_acc)):

        print("|----{}th round Test Accuracy : {:.2f}%".format(i,100*final_test_acc[i]))
    for i in range(len(final_backdoor_acc)):
        print("|----{}th round Backdoor Success Rate : {:.2f}%".format(i, 100 * final_backdoor_acc[i]))


    exp_details(args)
    file_n = f'accuracy_sync_{args.update_rule}_{args.dataset}_backdoor_scaled_{args.backdoor_scale}_{args.seed}.csv'

    f = open(file_n, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    for i in range((len(final_test_acc))):
        wr.writerow([i + 1, final_test_acc[i] * 100, final_backdoor_acc[i] * 100])

    f.close()












