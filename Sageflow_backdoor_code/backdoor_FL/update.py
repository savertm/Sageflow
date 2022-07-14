import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from utils2 import get_poison_batch
from options import args_parser


args = args_parser()

class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        self.dataset = dataset


        self.idxs = [int(i) for i in idxs]


    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)



class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, idx,data_poison, delay=False):
        self.args = args
        self.idx= idx

        self.trainloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'

        self.criterion = nn.NLLLoss().to(self.device)
        self.delay = delay
        self.data_poison = data_poison

    def train_val_test(self, dataset, idxs):



        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_test = idxs[int(0.8*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=max(int(len(idxs_test)/10),1), shuffle=False)

        return trainloader, testloader

    def update_weights(self,model, global_round,local_ep, backdoor=False ):
        model.train()
        epoch_loss = []

        if self.args.optimizer == 'sgd':

            lr = self.args.lr

            if backdoor == False:
                lr = 0.01
            elif backdoor == True and self.args.ignore_straggler==True:
                if args.dataset == 'cifar':
                    lr = 0.01 * (0.5) ** (global_round // 1400)
                else:
                    lr = 0.01 * (0.5)**(global_round//30)
            else:
                lr = 0.01

            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

        elif self.args.optimizer == 'adam':

            lr = self.args.lr
            lr = lr * (0.5)**(global_round//4)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        for iter in range(local_ep):
            batch_loss = []
            for batch_idx, (images_ori, labels_ori) in enumerate(self.trainloader):
                if backdoor==True:
                    images, labels = get_poison_batch((images_ori, labels_ori), self.args, self.device)
                else:
                    images, labels = images_ori.to(self.device), labels_ori.to(self.device)

                model.zero_grad()
                log_probs,_ = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs,_ = model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

            accuracy = correct/total
        return accuracy, loss

def test_inference(args, model, test_dataset):

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)

    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)



    batch_losses = []
    batch_entropy = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            output, out = model(images)

            Information = F.softmax(out, dim=1) * F.log_softmax(out, dim=1)
            entropy  = -1.0 * Information.sum(dim=1)
            average_entropy = entropy.mean().item()
            batch_entropy.append(average_entropy)

            batch_loss = criterion(output, labels)
            batch_losses.append(batch_loss.item())

            _, pred_labels = torch.max(output,1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)


        accuracy  = correct/total

    return accuracy, sum(batch_losses)/len(batch_losses), sum(batch_entropy)/len(batch_entropy)


def backdoor_test_inference(args, model, test_dataset):

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = f'cuda:{args.gpu_number}' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)

    testloader = poison_test_dataset(test_dataset)
    batch_losses = []
    batch_entropy = []
    with torch.no_grad():
        for batch_idx, (images_ori, labels_ori) in enumerate(testloader):
            images, labels = get_poison_batch((images_ori, labels_ori), args, device,evaluation=True)
            output, out = model(images)

            Information = F.softmax(out, dim=1) * F.log_softmax(out, dim=1)
            entropy  = -1.0 * Information.sum(dim=1)
            average_entropy = entropy.mean().item()
            batch_entropy.append(average_entropy)

            batch_loss = criterion(output, labels)
            batch_losses.append(batch_loss.item())

            _, pred_labels = torch.max(output,1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)


        accuracy  = correct/total

    return accuracy, sum(batch_losses)/len(batch_losses), sum(batch_entropy)/len(batch_entropy)


def poison_test_dataset(test_dataset):
    test_classes = {}
    for ind, x in enumerate(test_dataset):
        _, label = x

        if label in test_classes:
            test_classes[label].append(ind)
        else:
            test_classes[label] = [ind]

    range_no_id = list(range(0, len(test_dataset)))
    for image_ind in test_classes[2]:
        if image_ind in range_no_id:
            range_no_id.remove(image_ind)


    return DataLoader(test_dataset, batch_size=args.local_bs, sampler=torch.utils.data.sampler.SubsetRandomSampler(range_no_id))