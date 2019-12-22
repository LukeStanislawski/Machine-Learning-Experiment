import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import load_data, imshow
from net import FCN



class RunTest():
    def __init__(self, net, params, trainloader, testloader, id="Net"):
        self.id = id
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.res = {}
        self.res["params"] = params

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.res["params"]["lr"], momentum=self.res["params"]["momentum"])

        print("\nID: {}".format(self.id))
        self.main()

    def main(self):
        self.train()
        self.save(self.id)
        # self.load("FCN")
        self.test()
        self.save_results()


    def train(self):
        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0


    def test(self):
        # outputs = self.net(images)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

        class_correct = [0. for i in range(10)]
        class_total = [0. for i in range(10)]
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1


        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                self.res["params"]["data"]["classes"][i], 100 * class_correct[i] / class_total[i]))


    def load(self, name):
        path = "models/{}.pth".format(name)
        self.net.load_state_dict(torch.load(path))
        print("Model loaded from " + path)

    def save(self, name):
        path = "models/{}.pth".format(name)
        torch.save(self.net.state_dict(), path)
        print("Model saved to " + path)

    def save_results(self):
        with open('results.csv', 'a+') as f:
            f.write(json.dumps(self.res) + "\n")



def main():
    PATH = 'models/net2.pth'

    trainset, trainloader, testset, testloader, data_info = load_data()

    params = {}
    params["lr"] = 0.001
    params["momentum"] = 0.9
    params["data"] = data_info
    params["model"] = "FCN"

    for hl in range(4):
        params["hidden_layers"] = hl
        net = FCN(hidden_layers=hl)
        RunTest(net, params, trainloader, testloader, id="FCN{}".format(hl))

    # torch.save(net.state_dict(), PATH)


if __name__ == "__main__":
    main()
