import sys, os, time, datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
from scipy import stats


class RunTest():
    def __init__(self, X_train, y_train, X_test, y_test, folds=10, kernel='linear', C=1, ID="0", pca=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.folds = folds
        self.kernel = kernel
        self.C = C
        self.ID = ID
        self.pca = pca
        self.pca_time = 0
        self.datetime = datetime.datetime.now().strftime("%x %X")
        self.train_time = None
        self.runtime = None
        self.train_results = []
        self.test_results = None
        self.svc = SVC(kernel=self.kernel, C=self.C, gamma='auto')

        print("\nKernel: {}, C: {}, Folds: {}, PCA: {}".format(self.kernel, self.C, self.folds, self.pca))

        self.main()


    def main(self):
        tstart = time.time()

        self.apply_pca()
        tr_start = time.time()
        self.train()
        self.train_time = time.time() - tr_start
        self.test_results = self.test(self.X_test, self.y_test)
        
        self.runtime = time.time() - tstart
        self.log_results()


    def apply_pca(self):
        if self.pca is not None and int(self.pca) != 0:
            tstart = time.time()

            max_pca = min(len(self.X_train), len(self.X_train[0]))
            n = int(self.pca * max_pca)

            pca = PCA(n_components=n) #New dimentionality of feature vectors
            self.X_train = pca.fit_transform(self.X_train)
            self.X_test = pca.transform(self.X_test)

            self.pca_time = time.time() - tstart


    def train(self):      
        for i in range(self.folds):
            X_folds = np.array_split(self.X_train, self.folds)
            y_folds = np.array_split(self.y_train, self.folds)
            X_val = X_folds.pop(i)
            y_val = y_folds.pop(i)
            X_train = np.concatenate(X_folds)
            y_train = np.concatenate(y_folds)

            it_res={}
            it_res["iter"] = i
            it_res["datetime"] = datetime.datetime.now().strftime("%x %X")
            it_res["n_train"] = len(X_train)
            
            tstart = time.time()
            self.svc.fit(X_train, y_train)
            tend = time.time()
            
            it_res["t_train"] = tend - tstart
            it_res.update(self.test(X_val, y_val))
            
            self.train_results.append(it_res)
            print("{0} - f1: {1:.3f}, accuracy: {2:.3f}, runtime: {3:.3f}".format(i, it_res['f1'], it_res['accuracy'], it_res["runtime"]))


    def test(self, X, y):
        tstart = time.time()
        y_pred = self.svc.predict(X)
        t_test = time.time() - tstart

        test_res={}
        test_res["n_test"] = len(X)
        test_res["runtime"] = t_test
        test_res['f1'] = f1_score(y, y_pred, average='micro')
        test_res['accuracy'] = accuracy_score(y, y_pred)
        return test_res


    def log_results(self):
        o = [self.ID,
            self.datetime,
            self.kernel,
            self.C,
            self.folds,
            self.pca,
            "{0:.3f}".format(self.test_results["f1"]),
            "{0:.3f}".format(self.test_results["accuracy"]),
            len(self.X_train),
            len(self.X_test),
            "{0:.3f}".format(self.runtime),
            "{0:.3f}".format(self.pca_time),
            "{0:.3f}".format(self.train_time),
            "{0:.3f}".format(self.test_results["runtime"])]
        o_str = [str(x) for x in o]

        with open('results1.csv', 'a+') as f:
            f.write(",".join(o_str) + "\n")

        print("F - f1: {0:.3f}, accuracy: {1:.3f}, runtime: {2:.3f}".format(self.test_results["f1"], self.test_results["accuracy"], self.runtime))



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# def display_img(img):
#     img2 = [[img[x], img[1024+x], img[2048+x]] for x in range(int(len(img)/3))]  
#     img3 = [img2[x*32:(x+1)*32] for x in range(32)]
#     plt.imshow(img3)
#     plt.show()


def load_data():
    X_full = []
    y_full = []
    batch_labels = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]
    
    for batch_label in batch_labels:
        data = unpickle("cifar-10-batches-py/{}".format(batch_label))
        X_full.extend(data[b"data"])
        y_full.extend(data[b"labels"])

    return X_full, y_full


def main():
    print("Loading data")
    X_full, y_full = load_data()
    X_small = X_full[:500]
    y_small = y_full[:500]
    X_test = X_full[-50:]
    y_test = y_full[-50:]

    # X_small = [[n/255 for n in x] for x in X_small]
    # X_test = [[n/255 for n in x] for x in X_test]

    # print("Testing plain data..")
    # RunTest(X_small, y_small, X_test, y_test)

    # print("Testing PCA data..")
    # RunTest(X_small, y_small, X_test, y_test, pca=0.8)

    for pca in [0, 0.8, 0.5]:
        for kernel in ['linear', 'poly']:
            RunTest(X_small, y_small, X_test, y_test, kernel=kernel, pca=pca)


if __name__ == "__main__":
    main()