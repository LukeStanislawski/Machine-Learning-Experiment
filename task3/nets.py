import torch.nn as nn
import torch.nn.functional as F
# from torch.nn import Parameter as P

def get_nets():
    return [Net(), FCN()]


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


class FCN(nn.Module):
    def __init__(self, hidden_layers=1):
        super(FCN, self).__init__()

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



class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.mPool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(8, 24, 5)
        self.fc1 = nn.Linear(24 * 5 * 5, 300)
        self.fc2 = nn.Linear(300, 128)
        self.fc3 = nn.Linear(128, 10)

        self.p = False

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mPool(x)
        x = F.relu(self.conv2(x))
        x = self.mPool(x)
        x = x.view(-1, 24 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# USED IN REPORT
# --------------

class FCN2(nn.Module):
    def __init__(self, hidden_layers=1):
        super(FCN2, self).__init__()
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
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.fc1 = nn.Linear(6 * 28 * 28, 2048)
        self.fc2 = nn.Linear(2048, 10)

        self.p = False

    def forward(self, x):
        x = self.conv1(x)
        # print (x.size())
        x = x.view(-1, 6 * 28 * 28)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Conv2(nn.Module):
    def __init__(self):
        super(Conv2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.fc1 = nn.Linear(12 * 28 * 28, 2048)
        self.fc2 = nn.Linear(2048, 10)

        self.p = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 18 * 28 * 28)
        x = self.fc1(x)
        x = self.fc2(x)
        return x



class Conv3(nn.Module):
    def __init__(self):
        super(Conv3, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.conv3 = nn.Conv2d(12, 18, 3)
        self.fc1 = nn.Linear(18 * 28 * 28, 2048)
        self.fc2 = nn.Linear(2048, 10)

        self.p = False

    def forward(self, x):
        x = self.conv1(x)
        # print (x.size())
        x = self.conv2(x)
        # print (x.size())
        x = self.conv3(x)
        x = x.view(-1, 18 * 28 * 28)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# Docs
# ____

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, 
#   stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

# torch.nn.MaxPool2d(kernel_size, 
#   stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

# torch.nn.Linear(in_features, out_features, bias=True)


# print (x.size())