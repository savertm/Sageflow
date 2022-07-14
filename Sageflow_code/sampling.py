

import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):

        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))


        all_idxs = list(set(all_idxs) - dict_users[i])


    return dict_users





def mnist_noniid(dataset, num_users):


    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]


    idxs = idxs_labels[0,:]



    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard)- rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users



def mnist_noniid_unequal(dataset, num_users):


    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i:np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]



    min_shard = 1
    max_shard = 30

    random_shard_size = np.random.randint(min_shard, max_shard+1, size= num_users)

    random_shard_size = np.around(random_shard_size / sum(random_shard_size)*num_shards)

    random_shard_size = random_shard_size.astype(int)

    if sum(random_shard_size) > num_shards:
        for i in range(num_users):

            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

        random_shard_size = random_shard_size -1


        for i in range(num_users):
            if len(idx_shard) ==0:
                continue
            shard_size = random_shard_size[i]


            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    else:
        for i in range(num_users):
            shard_size = random_shard_size[i]

            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)


            if len(idx_shard)>0:

                shard_size = len(idx_shard)


                k = min(dict_users, key=lambda x: len(dict_users.get(x)))
                rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    dict_users[k] = np.concatenate((dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    return dict_users


def cifar_iid(dataset, num_users):


    num_items = int(len(dataset)/num_users)

    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False ))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_noniid(dataset, num_users):

    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i:np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)

    b = []
    for i in range(len(dataset)):
        b.append(dataset[i][1])


    labels = np.array(b)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]



    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    return dict_users

if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))]))
    num = 100
    d = mnist_noniid(dataset_train, num)




















