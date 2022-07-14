from torch import nn
import torch.nn.functional as F




class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return F.log_softmax(x,dim=1)



class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()

        self.conv1 = nn.Conv2d(1,32,5, padding=2)

        self.conv2 = nn.Conv2d(32,64,5, padding=2)

        self.fc1 = nn.Linear(64*7*7, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))

        x = self.fc2(x)


        return F.log_softmax(x, dim=1), x





class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1), out









class VGGCifar(nn.Module):

    def __init__(self):
        super(VGGCifar, self).__init__()
        #VGG11
        self.vgg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        self.features = self._make_layers(self.vgg)
        self.classifier = nn.Linear(512,10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return F.log_softmax(out, dim=1), out

    def _make_layers(self, vgg):
        layers = []
        in_channels = 3
        for x in vgg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           #nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels= x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)







class CNNCifar(nn.Module):
    def __init__(self,args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.conv5 = nn.Conv2d(64,64,3,padding=1)

        self.conv6 = nn.Conv2d(64,128,3,padding=1)
        self.conv7 = nn.Conv2d(128,128,3, padding=1)
        self.conv8 = nn.Conv2d(128,128,3,padding=1)
        self.fc1 = nn.Linear(128,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.avg_pool2d(x,8)
        x = x.view(-1,128)

        x = self.fc1(x)

        return F.log_softmax(x, dim=1)









