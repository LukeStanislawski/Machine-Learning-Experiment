import sys, os, logging
import torch
import torchvision
import torchvision.transforms as transforms
from  torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt


def load_data():
	transform = transforms.Compose(
	    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR10(root='../', train=True,
	                                        download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
	                                          shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='../', train=False,
	                                       download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=4,
	                                         shuffle=True, num_workers=2)
	info = {}
	info["n_train"] = len(trainset)
	info["n_test"] = len(testset)
	info["classes"] = get_classes()
	info["batch_size"] = 4

	return trainset, trainloader, testset, testloader, info


def get_classes():
	return ('plane', 'car', 'bird', 'cat',
	           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def get_label(index):
	return get_classes()[index]


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_logger(path):
	log = logging.getLogger("file_out")
	hdlr = logging.FileHandler(path)
	formatter = logging.Formatter('"%(asctime)s [%(levelname)-5.5s]  %(message)s"')
	hdlr.setFormatter(formatter)
	log.addHandler(hdlr) 
	log.setLevel(logging.INFO)
	logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
	return log


def silence_warnings():
    def warn(*args, **kwargs):
    	pass
    import warnings
    warnings.warn = warn


# Format time float to string
def ft(seconds):
	m, s = divmod(seconds, 60)
	h, m = divmod(m, 60)
	return "{:.0f}:{:.0f}:{:.0f}".format(h, m, s)


# def display_img(img):
#     img2 = [[img[x], img[1024+x], img[2048+x]] for x in range(int(len(img)/3))]  
#     img3 = [img2[x*32:(x+1)*32] for x in range(32)]
#     plt.imshow(img3)
#     plt.show()