import torch.nn as nn
import torch.nn.functional as F
# from torch.nn import Parameter as P

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.mPool = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.mPool(F.relu(self.conv1(x)))
        x = self.mPool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FCN_m(nn.Module):
    def __init__(self, hidden_layers=1):
        super(FCN_m, self).__init__()

        max_n = 32 * 32 * 3
        step = int( (max_n -10) / (hidden_layers+1) )
        n = max_n

        self.fc_hidden = []
        # self.fc_first = nn.Linear(max_n, n)
        for i in range(hidden_layers):
            self.fc_hidden.append(nn.Linear(n, n - step))
            n = n - step
        self.fc_last = nn.Linear(n, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        for fc in self.fc_hidden:
            x = fc(x)
        x = self.fc_last(x)
        return x


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.mPool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.mPool(F.relu(self.conv1(x)))
        x = self.mPool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# USED IN REPORT
# --------------

class FCN(nn.Module):
    def __init__(self, hidden_layers=1):
        super(FCN, self).__init__()
        self.fc_hidden = []
        max_n = 32 * 32 * 3
        for i in range(hidden_layers):
            self.fc_hidden.append(nn.Linear(max_n, max_n))
        self.fc_last = nn.Linear(max_n, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        for fc in self.fc_hidden:
            x = fc(x)
        x = self.fc_last(x)
        return x


class Conv1(nn.Module):
    def __init__(self):
        super(Conv1, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.mPool = nn.MaxPool2d(2,1)
        self.fc1 = nn.Linear(8 * 28 * 28, 2 * 28 * 28)
        self.fc2 = nn.Linear(2 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv1(x)
        # print (x.size())
        x = x.view(-1, 8 * 28 * 28)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Conv2(nn.Module):
    def __init__(self):
        super(Conv2, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 24, 5)
        self.mPool = nn.MaxPool2d(2,1)
        self.fc1 = nn.Linear(24 * 22 * 22, 2048)
        self.fc2 = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mPool(x)
        x = self.conv2(x)
        x = self.mPool(x)
        x = x.view(-1, 24 * 22 * 22)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Conv3(nn.Module):
    def __init__(self):
        super(Conv3, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 24, 5)
        self.conv3 = nn.Conv2d(24, 72, 5)
        self.mPool = nn.MaxPool2d(2,1)
        self.fc1 = nn.Linear(72 * 16 * 16, 2048)
        self.fc2 = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mPool(x)
        x = self.mPool(x)
        x = self.conv2(x)
        x = self.mPool(x)
        x = self.conv3(x)
        x = self.mPool(x)
        x = x.view(-1, 72 * 16 * 16)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class AvgPool(nn.Module):
    def __init__(self):
        super(AvgPool, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 24, 5)
        self.mPool = nn.AvgPool2d(2,1)
        self.fc1 = nn.Linear(24 * 22 * 22, 2048)
        self.fc2 = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mPool(x)
        x = self.conv2(x)
        x = self.mPool(x)
        x = x.view(-1, 24 * 22 * 22)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 24, 5)
        self.mPool = nn.MaxPool2d(2,1)
        self.fc1 = nn.Linear(24 * 22 * 22, 2048)
        self.fc2 = nn.Linear(2048, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mPool(x)

        x = F.relu(self.conv2(x))
        x = self.mPool(x)

        x = x.view(-1, 24 * 22 * 22)
        x = F.relu(self.fc1(x))     
        x = self.fc2(x)
        return x


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 24, 5)
        self.mPool = nn.MaxPool2d(2,1)
        self.fc1 = nn.Linear(24 * 22 * 22, 2048)
        self.fc2 = nn.Linear(2048, 10)

    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = self.mPool(x)

        x = F.sigmoid(self.conv2(x))
        x = self.mPool(x)

        x = x.view(-1, 24 * 22 * 22)
        x = F.sigmoid(self.fc1(x))     
        x = self.fc2(x)
        return x


class SoftMax(nn.Module):
    def __init__(self):
        super(SoftMax, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 24, 5)
        self.mPool = nn.MaxPool2d(2,1)
        self.fc1 = nn.Linear(24 * 22 * 22, 2048)
        self.fc2 = nn.Linear(2048, 10)

    def forward(self, x):
        x = F.softmax(self.conv1(x))
        x = self.mPool(x)

        x = F.softmax(self.conv2(x))
        x = self.mPool(x)

        x = x.view(-1, 24 * 22 * 22)
        x = F.softmax(self.fc1(x))     
        x = self.fc2(x)
        return x


class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 24, 5)
        self.mPool = nn.MaxPool2d(2,1)
        self.fc1 = nn.Linear(24 * 22 * 22, 2048)
        self.fc2 = nn.Linear(2048, 10)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.mPool(x)

        x = F.tanh(self.conv2(x))
        x = self.mPool(x)

        x = x.view(-1, 24 * 22 * 22)
        x = F.tanh(self.fc1(x))     
        x = self.fc2(x)
        return x


class ReLU6(nn.Module):
    def __init__(self):
        super(ReLU6, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 24, 5)
        self.mPool = nn.MaxPool2d(2,1)
        self.fc1 = nn.Linear(24 * 22 * 22, 2048)
        self.fc2 = nn.Linear(2048, 10)

    def forward(self, x):
        x = F.relu6(self.conv1(x))
        x = self.mPool(x)

        x = F.relu6(self.conv2(x))
        x = self.mPool(x)

        x = x.view(-1, 24 * 22 * 22)
        x = F.relu6(self.fc1(x))     
        x = self.fc2(x)
        return x


class Channel(nn.Module):
    def __init__(self, chan2, chan3):
        super(Channel, self).__init__()
        self.chan3 = chan3
        self.conv1 = nn.Conv2d(3, chan2, 5)
        self.conv2 = nn.Conv2d(chan2, chan3, 5)
        self.mPool = nn.MaxPool2d(2,1)
        self.fc1 = nn.Linear(chan3 * 22 * 22, 2048)
        self.fc2 = nn.Linear(2048, 10)

    def forward(self, x):
        x = F.relu6(self.conv1(x))
        x = self.mPool(x)
        x = F.relu6(self.conv2(x))
        x = self.mPool(x)
        x = x.view(-1, self.chan3 * 22 * 22)
        x = F.relu6(self.fc1(x))     
        x = self.fc2(x)
        return x


class HyperParams(nn.Module):
    def __init__(self):
        super(HyperParams, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.mPool = nn.MaxPool2d(2,1)
        self.fc1 = nn.Linear(20 * 22 * 22, 2048)
        self.fc2 = nn.Linear(2048, 10)

    def forward(self, x):
        x = F.relu6(self.conv1(x))
        x = self.mPool(x)
        x = F.relu6(self.conv2(x))
        x = self.mPool(x)
        x = x.view(-1, 20 * 22 * 22)
        x = F.relu6(self.fc1(x))     
        x = self.fc2(x)
        return x