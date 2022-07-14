import copy
import torch
from torchvision import datasets, transforms

import sys

from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling_withcommon import mnist_noniidcmm, cifar_noniidcmm
from update import LocalUpdate, test_inference, DatasetSplit
from math import exp
import numpy as np
from numpy import linalg
from options import args_parser
import math
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor,Resize,Normalize
import pdb

def get_dataset(args):

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)

        if args.iid == 1:

            user_groups = cifar_iid(train_dataset, args.num_users)

        elif args.iid == 2:

            user_groups = cifar_noniidcmm(train_dataset, args.num_users, args.num_commondata)

        else:
            if args.unequal:
                raise NotImplementedError()
            else:
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if args.dataset == 'mnist':
            data_dir = '../data/mnist'
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)
        else:
            data_dir = '../data/fmnist'
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                                  transform=apply_transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                                 transform=apply_transform)

        if args.iid == 1:

            user_groups = mnist_iid(train_dataset, args.num_users)

        elif args.iid == 2:

            user_groups = mnist_noniidcmm(train_dataset, args.num_users, args.num_commondata)

    return train_dataset, test_dataset, user_groups


# Federated averaging

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def compute_gradient_norm(w1, w2):
    w_avg = copy.deepcopy(w1)
    norm = 0
    for key in w_avg.keys():

        w_avg[key] = w1[key] - w2[key]
        norm += torch.norm(w_avg[key])
    return norm

# Loss_weighted_average
def weighted_average(w, beta):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(0, len(w)):
            if i == 0:
                w_avg[key] = w[i][key] * beta[0]
            else:
                w_avg[key] += w[i][key] * beta[i]
    return w_avg




# Staleness-aware grouping

def Sag(current_epoch, current_average, current_length, epoch_weights, global_weights):
    args = args_parser()

    alpha = []
    weights_d = []
    num_device = []

    alpha_for_attack = np.ones(args.staleness+1)

    comm = current_length
    for i in epoch_weights:
        key = list(i.keys())[0]
        alpha.append(key)
        weights_d.append(i[key][0])
        num_device.append(i[key][1])
        comm = comm + i[key][1]

    # For empty staleness groups
    # You can ignore this part
    #########################################################################################
    if current_average is not None:
        w_semi = copy.deepcopy(current_average)

    else:
        for weigts_delay in weights_d:
            if weigts_delay is not None:
                w_semi = copy.deepcopy(weigts_delay)
                break

    if current_epoch >=args.staleness:
        if current_average is None:
            alpha_for_attack[0] = 0

        for i in range(args.staleness + 1):
            if i != 0 and weights_d[i-1] is None:
                alpha_for_attack[i] = 0

    #########################################################################################

    #Staleness-based weight
    alphas = (current_epoch - np.array(alpha) + 1) ** (-args.lam)

    alphas = alphas * np.array(num_device)
    alphas = alphas * alpha_for_attack[1:len(alpha) + 1]


    if len(alphas) == 0:
        alphas = np.array([current_length * alpha_for_attack[0]])
    else:
        alphas = np.concatenate((np.array([current_length * alpha_for_attack[0]]), alphas), axis=0)

    sum_alphas = sum(alphas)
    alphas = alphas / sum_alphas


    for key in w_semi.keys():
        for i in range(0, len(weights_d) + 1):
            if i == 0:
                w_semi[key] = w_semi[key] * (alphas[0])
            else:
                if weights_d[i-1] is None:
                    continue
                else:
                    w_semi[key] += weights_d[i - 1][key] * alphas[i]


    for key in w_semi.keys():
        if args.dataset =='cifar':
            alpha = 0.5 ** (current_epoch // 300)
        elif args.dataset =='fmnist':
            alpha = 0.5 ** (current_epoch // 20)

        elif args.dataset =='mnist':
            alpha = 0.5 ** (current_epoch // 15)

        w_semi[key] = w_semi[key] * (alpha) + global_weights[key] * (1 - alpha)

    return w_semi



def communication_w(w, w_pre):

    w_com = copy.deepcopy(w)

    for key in w_com.keys():
        w_com[key] = w[key] - w_pre[key]

    return w_com

def receive_w(w, w_pre):

    w_com = copy.deepcopy(w)

    for key in w_com.keys():
        w_com[key] = w[key] + w_pre[key]

    return w_com




# Entropy based filtering and loss weighted averaging

def Eflow(w, loss, entropy, current_epoch, num_device=[]):

    args=args_parser()
    w_avg = copy.deepcopy(w[0])
    num_attack = 0
    alpha = []

    for j in range(0, len(loss)):

        if entropy[j] >= args.eth:
            norm_q = 0
            num_attack += 1
        else:
            norm_q = 1

        if len(num_device) == 0:
            alpha.append(norm_q / loss[j] ** args.delta)


    sum_alpha = sum(alpha)

    if sum_alpha <= 0.001:
        for k in range(0, len(alpha)):
            w_avg = None

    else:
        for k in range(0, len(alpha)):
            alpha[k] = alpha[k] / sum_alpha

        for key in w_avg.keys():
            for i in range(0, len(w)):
                if i == 0:
                    w_avg[key] = w_avg[key] * alpha[i]

                else:
                    w_avg[key] += w[i][key] * alpha[i]

    return w_avg, len(loss) - num_attack




def sign_attack(w,scale=0.1):
    w_avg = copy.deepcopy(w)
    for key in w_avg.keys():
        w_avg[key] = -w[key] * scale
    return w_avg




def exp_details(args):
    print('\nExperimental details:')
    print(f'    Dataset     : {args.dataset}')
    print(f'    Model     : {args.model}')
    print(f'    detailed Model     : {args.detail_model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid == 1:
        print('    IID')
    elif args.iid == 2:
        print('    Non-IID with common data')


    else:
        print('    Non-IID')
    if args.unequal:
        print('    Unbalanced')
    else:
        print('    balanced')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')

    print(f'    Attack ratio : {args.attack_ratio}')
    if args.data_poison == True:
        print('     Data poison attack is done!')
    elif args.model_poison == True:
        print('     Model attack is done!')
    else:
        print('     None of attack is done!\n')

    return





