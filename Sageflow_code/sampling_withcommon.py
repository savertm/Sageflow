import numpy as np
from torchvision import datasets, transforms




# Split the entire data into public data and users' data

def mnist_noniidcmm(dataset, num_users, num_commondata):

    
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users = {i: np.array([]) for i in range(1, num_users+1)}

    dict_users[0] = set(np.random.choice(all_idxs, num_commondata, replace=False))

    #Exclude the public data from local device
    all_idxs = list(set(all_idxs) - dict_users[0])
    total_data = len(all_idxs)
    num_shards, num_imgs = 200, total_data//200

    '''
    #include the public data
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    '''


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

    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i:np.array([]) for i in range(num_users+1)}
    idxs = np.arange(num_shards * num_imgs)
    dict_users[0] = set(np.random.choice(idxs, num_commondata, replace=False))

    # Exclude the public data from local device
    idxs = list(set(idxs) - dict_users[0])
    total_data = len(idxs)
    num_shards, num_imgs = 200, total_data//200

    b = []
    for i in idxs:
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










if __name__ == '__main__':
    if __name__ == '__main__':
        dataset_train = datasets.MNIST('../data/mnist', train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
