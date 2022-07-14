import argparse

import argparse


# Reference
# The original backbone code comes from
# https://github.com/AshwinRJ/Federated-Learning-PyTorch/tree/master/src



def args_parser():
    parser = argparse.ArgumentParser()

    # Training parameter
    parser.add_argument('--epochs', type=int, default=300,
                        help="number of training rounds")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=5,
                        help="number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--staleness', type=int, default=2,
                        help='maximum staleness)')
    parser.add_argument('--update_rule', type=str, default='Sageflow',
                        help='choose Sageflow or Fedavg')

    # The amount of Public data
    parser.add_argument('--num_commondata', type=float, default=1000,
                        help='number of public data which server has')


    # For Adversarial attack
    parser.add_argument('--attack_ratio', type=float, default=0.1,
                        help='attack ratio')
    parser.add_argument('--num_img_backdoor', type=float, default=20,
                        help='number of poisoned images in a batch')
    parser.add_argument('--data_poison', type=str2bool, default=True,
                        help='True: data poisoning attack (backdoor attack), False: no attack')
    parser.add_argument('--backdoor_scale', type=float, default=10,
                        help='scaled (10) or no scaled (1) backdoor attack')
    parser.add_argument('--model_poison', type=str2bool, default=False,
                        help='True: model poisoning attack, False: no attack')

    # Hyperparameters of Sageflow
    parser.add_argument('--eth', type=float, default=2.5,
                        help='Eth of Eflow')
    parser.add_argument('--delta', type=float, default=5,
                        help='Delta of Eflow')
    parser.add_argument('--lam', type=float, default=0.5,
                        help='lambda of Sag')

    # Other arguments
    parser.add_argument('--dataset', type=str, default='fmnist', help="name \
                        of dataset: choose mnist or fmnist or cifar")
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--gpu', default=True, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--gpu_number', default=6, help="GPU number to use")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                            of classes")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")

    # Data distribution setting
    parser.add_argument('--iid', type=int, default=3,
                        help='Set to 3 for backdoor setting')

    # Detailed settings
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')

    # Model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--detail_model', type=str, default='vgg', help='model name')

    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                            use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                            of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                            mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='Teometricrue',
                        help="Whether use max pooling rather than \
                            strided convolutions")
    parser.add_argument('--ignore_straggler', type=str2bool, default='False',
                        help='Ignore straggler scheme setting:True')

    args = parser.parse_args()
    return args

def str2bool(v):
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False