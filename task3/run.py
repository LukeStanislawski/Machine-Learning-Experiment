import sys, os, json, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import load_data, imshow, get_label
from nets import FCN, FCN2, MainNet, Conv1, Conv2, Conv3
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



class RunTest():
    def __init__(self, net, params, trainloader, testloader, id=None):
        self.id = id
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.res = {}
        self.res["params"] = params
        self.res["params"]["n_train"] = len(trainloader)
        self.res["params"]["n_test"] = len(testloader)
        self.res["test"] = {}
        self.res["train"] = {}
        if id is None:
            self.id = "Net"
            self.res["params"] = self.id
        else:
            self.res["params"]["id"] = self.id



        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.res["params"]["lr"], momentum=self.res["params"]["momentum"])

        print("\nID: {}".format(self.id))
        self.main()


    def main(self):
        self.train()
        self.save(self.id)
        # self.load(self.id)
        self.res["test"] = self.test()
        self.save_results()


    def train(self):
        train_start = time.time()
        self.res["train"]["epochs"] = []

        for epoch in range(self.res["params"]["num_epochs"]):
            print("Epoch: {}".format(epoch+1))
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                if i <= 2000:
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
                        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

            self.res["train"]["epochs"].append(self.test())
        self.res["train"]["runtime"] = time.time() - train_start


    def test(self):
        test_res = {}
        test_start = time.time()
        y_test = []
        y_pred = []
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                y_test.extend(labels.tolist())
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.tolist())

        test_res["accuracy"] = float(accuracy_score(y_test, y_pred))
        test_res["f1_pc"] = list(f1_score(y_test, y_pred, labels=range(10), average=None))
        test_res["precision"] = precision_score(y_test, y_pred, average='micro')
        test_res["precision_pc"] = list(precision_score(y_test, y_pred, labels=range(10), average=None))
        test_res["recall"] = recall_score(y_test, y_pred, average='micro')
        test_res["recall_pc"] = list(recall_score(y_test, y_pred, labels=range(10), average=None))


        print ("Accuracy: {}".format(test_res["accuracy"]))
        for i, f1 in enumerate(test_res["f1_pc"]):
            print("{}: {:.3f}".format(get_label(i), f1))

        test_res["runtime"] = time.time() - test_start
        return test_res



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


def silence_warnings():
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn



# def test_FCN(params, trainloader, testloader, layers=[1,2,3]):
    



def main():
    trainset, trainloader, testset, testloader, data_info = load_data()

    params = {}
    params["lr"] = 0.001
    params["momentum"] = 0.9
    params["data"] = data_info
    params["num_epochs"] = 2


    params["model"] = "FCN"
    for layer in [1, 2, 3]:
        RunTest(FCN2(hidden_layers=layer), params, trainloader, testloader, id="FCN_{}".format(layer))


    params["model"] = "ConvX"
    RunTest(Conv1(), params, trainloader, testloader, id="Conv1")
    RunTest(Conv2(), params, trainloader, testloader, id="Conv2")
    RunTest(Conv3(), params, trainloader, testloader, id="Conv3")


if __name__ == "__main__":
    silence_warnings()
    main()
