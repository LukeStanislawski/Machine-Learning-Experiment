import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def main():
    data = unpickle("cifar-100-python/test")
    img = data[b"data"][5]
    img2 = [[img[x], img[1024+x], img[2048+x]] for x in range(int(len(img)/3))]  
    img3 = [img2[x*32:(x+1)*32] for x in range(32)]

    plt.imshow(img3)
    plt.show()


if __name__ == "__main__":
    main()