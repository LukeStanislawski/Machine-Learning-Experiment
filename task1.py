import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats

def myknn(sample, X_feat, Y_feat, k):
    euclids = []
    for i, fv in enumerate(X_feat):
        obj = {}
        obj["X"] = fv
        obj["Y"] = Y_feat[i]
        dist = [(sample[n] - fv[n]) ** 2 for n in range(len(sample))]
        obj["dist"] = np.sqrt(sum(dist))
        euclids.append(obj)

    euclids = sorted(euclids, key = lambda i: i['dist'])
    knn = [euclids[i]["Y"] for i in range(k)]

    print (knn)

    return stats.mode(knn).mode[0]


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def display_img(img):
    img2 = [[img[x], img[1024+x], img[2048+x]] for x in range(int(len(img)/3))]  
    img3 = [img2[x*32:(x+1)*32] for x in range(32)]
    plt.imshow(img3)
    plt.show()


def test(X_test, y_test, X_train, y_train):
    correct = 0
    tested = 0
    for i in range(len(X_test)):
        res = myknn(X_test[i], X_train, y_train, 4)
        if res == y_test[i]:
            res_str = "Correct  "
            correct += 1
        else:
            res_str = "Incorrect"

        tested += 1
        print ("{}/{} - {}: pred: {} actual: {}".format(correct, tested, res_str, res, y_test[i]))


def main():
    data = unpickle("cifar-100-python/test")
    # data[b'filenames', b'batch_label', b'fine_labels', b'coarse_labels', b'data']

    X_small = data[b"data"][:100]
    y_small = data[b'fine_labels'][:100]
    X_test = data[b"data"][-50:]
    y_test = data[b'fine_labels'][-50:]

    print (len(X_test))

    test(X_test, y_test, X_small, y_small)


    # pca = PCA(n_components=8000)
    # X_small_trans = pca.fit_transform(X_small)
    # X_test_trans = pca.transform(X_test)

    # print(X_small_trans[0])



if __name__ == "__main__":
    main()