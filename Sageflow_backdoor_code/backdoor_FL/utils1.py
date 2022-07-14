import copy
import torch
from torchvision import datasets, transforms
import pdb
import sys



from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling_withcommon import mnist_noniidcmm, cifar_noniidcmm,mnist_dirichlet_sample, cifar_dirichlet_sample
from update import LocalUpdate, test_inference, DatasetSplit
from math import exp
import numpy as np
from numpy import linalg
from options import args_parser
import math


def get_dataset(args):

    if args.dataset == 'cifar':
        data_dir = '../../data/cifar/'
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

        # Backdoor setting
        elif args.iid == 3:
            user_groups = cifar_dirichlet_sample(train_dataset, args.num_users, args.num_commondata, alpha=0.5)

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
            data_dir = '../../data/mnist'
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)
        else:
            data_dir = '../../data/fmnist'
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                                  transform=apply_transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                                 transform=apply_transform)

        if args.iid == 1:
            user_groups = mnist_iid(train_dataset, args.num_users)

        elif args.iid == 2:

            user_groups = mnist_noniidcmm(train_dataset, args.num_users, args.num_commondata, args.num_recdata)

        # Backdoor setting
        elif args.iid == 3:
            user_groups = mnist_dirichlet_sample(train_dataset, args.num_users, args.num_commondata)


        else:
            if args.unequal:
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)

            else:
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


# Federated averaging
def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def weighted_average(w, beta):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(0, len(w)):
            if i == 0:
                w_avg[key] = w[i][key] * beta[0]
            else:
                w_avg[key] += w[i][key] * beta[i]
    return w_avg

def communication_w(w, w_pre, backdoor=False, gamma=1):

    w_com = copy.deepcopy(w)

    if backdoor==False:

        for key in w_com.keys():

            w_com[key] = w[key] - w_pre[key]
    else:

        for key in w_com.keys():

            w_com[key] = (w[key] - w_pre[key]) * gamma

    return w_com


def receive_w(w, w_pre):

    w_com = copy.deepcopy(w)

    for key in w_com.keys():

        w_com[key] = w[key] + w_pre[key]

    return w_com



def averge_weights_comm_eflw(ws, w_pre, global_model,args, train_dataset, dict_common, current_epoch, eta=1):

    loss_on_public =[]
    entropy_on_public = []
    for i in range(len(ws)):
        for key in ws[0].keys():
            ws[i][key] = ws[i][key] + w_pre[key]

        global_model.load_state_dict(ws[i])

        # Compute the loss and entropy for each device on public dataset
        common_acc, common_loss, common_entropy = test_inference(args, global_model,
                                                                          DatasetSplit(train_dataset, dict_common))
        loss_on_public.append(common_loss)
        entropy_on_public.append(common_entropy)

    # Entropy based filtering and loss weighted averaging
    w_avg, _ = Eflow(ws, loss_on_public, entropy_on_public, current_epoch, num_device=[])

    return w_avg


def compute_gradient_norm(w1, w2):
    w_avg = copy.deepcopy(w1)
    norm = 0
    for key in w_avg.keys():

        w_avg[key] = w1[key] - w2[key]
        norm += torch.norm(w_avg[key])
    return norm

#////////////////////////////backdoor attack///////////////////////////////////////
def get_poison_batch(batch, args, device, evaluation=False):

    images, targets = batch

    new_images = images
    new_targets = targets

    for index in range(0,len(images)):
        if evaluation:
            new_targets[index] = 2
            new_images[index] = add_pixel_pattern(images[index])

        else:
            if index < args.backdoor_ratio:
                new_targets[index] = 2
                new_images[index] = add_pixel_pattern(images[index])

            else:
                new_images[index] = images[index]
                new_targets[index] = targets[index]

    new_images = new_images.to(device)
    new_targets = new_targets.to(device).long()
    if evaluation:
        new_images.requires_grad_(False)
        new_targets.requires_grad(False)
    return new_images, new_targets

def add_pixel_pattern(image_ori):
    image = copy.deepcopy(image_ori)
    image[0][0][0] = 1

    return image


#////////////////////////////////////////////////////////////////////////////////
# Staleness aware grouping

def Sag(current_epoch, current_average, current_length, epoch_weights, global_weights, args):
    args = args_parser()


    alpha = []
    weights_d = []
    num_device = []

    alpha_for_attack = np.ones(args.staleness + 1)

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

    if current_epoch >= args.staleness:
        if current_average is None:
            alpha_for_attack[0] = 0

        for i in range(args.staleness + 1):
            if i != 0 and weights_d[i - 1] is None:
                alpha_for_attack[i] = 0

    #########################################################################################

    # Staleness-based weight
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
        alpha = 0.5 ** (current_epoch // 1000)
        w_semi[key] = w_semi[key] * (alpha) + global_weights[key] * (1 - alpha)

    return w_semi, comm



# Entropy gated loss weighted averaging
def Eflow(w, loss, entropy, current_epoch, num_device=[]):
    args = args_parser()
    w_avg = copy.deepcopy(w[0])
    num_attack = 0
    alpha = []

    for j in range(0, len(loss)):

        if entropy[j] >= args.eth:
            norm_q = 0
            num_attack += 1
        else:
            norm_q = 1

        alpha.append(norm_q / loss[j] ** args.delta)

    sum_alpha = sum(alpha)

    if sum_alpha <= 0.0001:
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

def average_weights_comm_fedavg(ws,w_pre,averaging_rule, eta=1 ):
    w_avg = copy.deepcopy(w_pre)

    w_trans = average_weights(ws)

    for key in w_avg.keys():

        w_avg[key] = w_pre[key] + eta * w_trans[key]

    return w_avg

def sign_attack(w):
    w_avg = copy.deepcopy(w)
    for key in w_avg.keys():
        w_avg[key] = -w[key]

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
        print(f'    Supplied data: {args.num_recdata}')

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





