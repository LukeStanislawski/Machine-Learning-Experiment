import sys, os, json, time, datetime, traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import load_data, imshow, get_label, silence_warnings, ft, get_logger

from nets import FCN
from nets import Conv1, Conv2, Conv3
from nets import AvgPool
from nets import ReLU, Sigmoid, SoftMax, Tanh, ReLU6
from nets import Channel
from nets import HyperParams


log = get_logger(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task3.log'))


class RunTest():
    def __init__(self, net, params, trainloader, testloader, id=None):
        self.id = id
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.res = {}
        self.res["datetime"] = datetime.datetime.now().strftime("%x %X")
        self.res["params"] = params
        self.res["test"] = {}
        self.res["train"] = {}
        if id is None:
            self.id = "Net"
            self.res["params"] = self.id
        else:
            self.res["params"]["id"] = self.id



        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.res["params"]["lr"], momentum=self.res["params"]["momentum"])

        print("\nID: {} - {}".format(self.id, self.res["datetime"]))
        self.main()


    def main(self):
        startime = time.time()
        try:
            # self.load(self.id)
            self.train()
            self.save(self.id)
            self.res["test"] = self.test()

            self.res["runtime"] = time.time() - startime
            print ("Runtime: {}".format(ft(self.res["runtime"])))
        except Exception as e:
            log.info("ERROR while running test")
            log.info(str(e))
            log.info(sys.exc_info())
            log.info(traceback.format_exc())
        self.save_results()


    def train(self):
        train_start = time.time()
        self.res["train"]["epochs"] = []

        for epoch in range(self.res["params"]["num_epochs"]):
            running_loss = 0.0
            data = iter(self.trainloader)
            for i in range(int(self.res["params"]["n_train"] / self.res["params"]["data"]["batch_size"])):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data.next()

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

            if self.res["params"]["test_every_epoch"]:
                self.res["train"]["epochs"].append(self.test())
                self.save_results()
        self.res["train"]["runtime"] = time.time() - train_start


    def test(self):
        print("Testing..")
        test_res = {}
        test_start = time.time()
        y_test = []
        y_pred = []
        data = iter(self.testloader)
        with torch.no_grad():
            for i in range(int(self.res["params"]["n_test"] / self.res["params"]["data"]["batch_size"])):
                images, labels = data.next()
                y_test.extend(labels.tolist())
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.tolist())

        test_res["accuracy"] = float(accuracy_score(y_test, y_pred))
        test_res["f1_pc"] = list(f1_score(y_test, y_pred, labels=range(10), average=None))
        test_res["f1"] = np.mean(test_res["f1_pc"])
        test_res["precision"] = precision_score(y_test, y_pred, average='micro')
        test_res["precision_pc"] = list(precision_score(y_test, y_pred, labels=range(10), average=None))
        test_res["recall"] = recall_score(y_test, y_pred, average='micro')
        test_res["recall_pc"] = list(recall_score(y_test, y_pred, labels=range(10), average=None))


        print ("Accuracy: {}".format(test_res["accuracy"]))
        # for i, f1 in enumerate(test_res["f1_pc"]):
        #     print("{}: {:.3f}".format(get_label(i), f1))

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
    


def main():
    trainset, trainloader, testset, testloader, data_info = load_data()

    params = {}
    params["data"] = data_info
    params["n_train"] = 50000
    params["n_test"] = 10000
    params["lr"] = 0.001
    params["momentum"] = 0.9
    params["num_epochs"] = 2
    params["test_every_epoch"] = False


    params["model"] = "FCN"
    params["tid"] = 0
    for layer in [1, 2, 3]:
        params["hidden_layers"] = layer
        # RunTest(FCN(hidden_layers=layer), params, trainloader, testloader, id="FCN_{}".format(layer))


    # params["model"] = "ConvX"
    # params["tid"] = 2
    # RunTest(Conv1(), params, trainloader, testloader, id="Conv1")
    # RunTest(Conv2(), params, trainloader, testloader, id="Conv2")
    # RunTest(Conv3(), params, trainloader, testloader, id="Conv3")


    # params["tid"] = 3
    # params["model"] = "AvgPool"
    # RunTest(AvgPool(), params, trainloader, testloader, id="AvgPool")


    # params["tid"] = 4
    # params["model"] = "ReLU"
    # RunTest(ReLU(), params, trainloader, testloader, id="ReLU")
    # params["model"] = "Sigmoid"
    # RunTest(Sigmoid(), params, trainloader, testloader, id="Sigmoid")
    # params["model"] = "SoftMax"
    # RunTest(SoftMax(), params, trainloader, testloader, id="SoftMax")
    # params["model"] = "Tanh"
    # RunTest(SoftMax(), params, trainloader, testloader, id="Tanh")
    # params["model"] = "ReLU6"
    # RunTest(ReLU6(), params, trainloader, testloader, id="ReLU6")


    # params["tid"] = 5
    # params["model"] = "Channel"
    # RunTest(Channel(5, 15), params, trainloader, testloader, id="(5, 15)")
    # RunTest(Channel(6, 18), params, trainloader, testloader, id="(6, 18)")
    # RunTest(Channel(10,20), params, trainloader, testloader, id="(10,20)")
    # RunTest(Channel(14,42), params, trainloader, testloader, id="(14,42)")


    # params["tid"] = 6
    # params["model"] = "HyperParams"
    # params["num_epochs"] = 12
    # params["test_every_epoch"] = True
    # RunTest(HyperParams(), params, trainloader, testloader, id="HyperParams_Epoch")


    params["tid"] = 7
    params["model"] = "HyperParams"
    params["num_epochs"] = 5
    params["test_every_epoch"] = False
    for lr_e in [-1, -2, -3, -4, -5]:
        params["lr"] = 10 ** lr_e
        RunTest(HyperParams(), params, trainloader, testloader, id="HyperParams_lr")


if __name__ == "__main__":
    silence_warnings()
    main()
