import numpy as np
from torchvision import datasets, transforms
from collections import defaultdict



# Split the entire data into public data and users' data

def mnist_noniidcmm(dataset, num_users, num_commondata, num_recdata):
    
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users = {i: np.array([]) for i in range(1,num_users+1)}

    total_data = len(all_idxs)


    dict_users[0] = set(np.random.choice(all_idxs, num_commondata, replace=False))


    # Exclude the public data from local device
    '''
    dict_users[0] = set(np.random.choice(all_idxs, num_commondata, replace=False))
    all_idxs = list(set(all_idxs) - dict_users[0])
    num_shards, num_imgs = 200, total_data//200
    '''

    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]


    labels = dataset.train_labels.numpy()

    idxs_labels = np.vstack((all_idxs, labels[all_idxs]))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

    idxs = idxs_labels[0,:]


    for i in range(1,num_users+1):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard)- rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    dict_common = dict_users[0]
    for i in range(num_users):
        dict_users[i] = dict_users[i+1]
    del dict_users[num_users]

    return dict_users, dict_common

def cifar_noniidcmm(dataset, num_users, num_commondata):
    """
        Sample non-I.I.D client data from CIFAR10 dataset
        :param dataset:
        :param num_users:
        :return:
        """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i:np.array([]) for i in range(num_users+1)}
    idxs = np.arange(num_shards * num_imgs)
    dict_users[0] = set(np.random.choice(idxs, num_commondata, replace=False))



    b = []
    for i in range(len(dataset)):
        b.append(dataset[i][1])


    labels = np.array(b)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]



    for i in range(1,num_users+1):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    dict_common = dict_users[0]

    for i in range(num_users):
        dict_users[i] = dict_users[i + 1]
    del dict_users[num_users]


    return dict_users, dict_common


# Sampling for backdoor

def build_class_idx(train_dataset):
    mnist_classes = {}
    for idx, data in enumerate(train_dataset):
        _, label = data
        if label in mnist_classes:
            mnist_classes[label].append(idx)
        else:
            mnist_classes[label] = [idx]
    return mnist_classes

def mnist_dirichlet_sample(dataset, num_users, num_commondata, alpha=0.5):

    all_idxs =  [i for i in range(len(dataset))]

    dict_users =defaultdict(list)
    dict_users[0] = set(np.random.choice(all_idxs, num_commondata, replace =False))

    mnist_classes = build_class_idx(dataset)
    class_size = len(mnist_classes[0])

    image_nums = []

    for c in range(10):
        image_num = []
        np.random.shuffle(mnist_classes[c])
        sampled_prob = class_size * np.random.dirichlet(
            np.array(num_users * [alpha])
        )
        for user in range(num_users):
            no_imgs = int(round(sampled_prob[user]))
            sampled_list = mnist_classes[c][:min(len(mnist_classes[c]), no_imgs)]
            image_num.append(len(sampled_list))
            dict_users[user+1].extend(sampled_list)
            mnist_classes[c] = mnist_classes[c][min(len(mnist_classes[c]), no_imgs):]

    dict_common = dict_users[0]

    for i in range(num_users):
        dict_users[i] = dict_users[i + 1]
    del dict_users[num_users]

    return dict_users, dict_common

def cifar_dirichlet_sample(dataset, num_users, num_commondata, alpha=0.5):

    all_idxs =  [i for i in range(len(dataset))]

    dict_users =defaultdict(list)
    dict_users[0] = set(np.random.choice(all_idxs, num_commondata, replace =False))

    mnist_classes = build_class_idx(dataset)
    class_size = len(mnist_classes[0])


    for c in range(10):
        image_num = []
        np.random.shuffle(mnist_classes[c])
        sampled_prob = class_size * np.random.dirichlet(
            np.array(num_users * [alpha])
        )
        for user in range(num_users):
            no_imgs = int(round(sampled_prob[user]))
            sampled_list = mnist_classes[c][:min(len(mnist_classes[c]), no_imgs)]
            image_num.append(len(sampled_list))
            dict_users[user+1].extend(sampled_list)
            mnist_classes[c] = mnist_classes[c][min(len(mnist_classes[c]), no_imgs):]

    dict_common = dict_users[0]

    for i in range(num_users):
        dict_users[i] = dict_users[i + 1]
    del dict_users[num_users]

    return dict_users, dict_common
















if __name__ == '__main__':
    if __name__ == '__main__':
        dataset_train = datasets.MNIST('../data/mnist', train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
