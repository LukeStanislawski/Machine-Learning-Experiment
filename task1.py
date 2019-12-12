import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
from scipy import stats


class Run():
    def __init__(self, X_train, y_train, X_test, y_test, folds=10, kernel='linear', C=1):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.folds = folds
        self.kernel = kernel
        self.C = C

        self.main()


    def main(self):
        self.svc = SVC(kernel=self.kernel, C=self.C)
        res = self.train()
        f_res = self.test(self.X_test, self.y_test, "F")
        res.append(f_res)
        return res


    def train(self): 
        scores = []       
        for i in range(self.folds):
            X_folds = np.array_split(self.X_train, self.folds)
            y_folds = np.array_split(self.y_train, self.folds)
            X_val = X_folds.pop(i)
            y_val = y_folds.pop(i)
            X_train = np.concatenate(X_folds)
            y_train = np.concatenate(y_folds)
            
            self.svc.fit(X_train, y_train)
            y_pred = self.svc.predict(X_val)
            scores.append(self.test(X_val, y_val, i))

        return scores


    def test(self, X, y, label):
        y_pred = self.svc.predict(X)

        res = {}
        res['f1'] = f1_score(y, y_pred, average='micro')
        res['accuracy'] = accuracy_score(y, y_pred)

        print("{0} - f1: {1}, accuracy: {2}".format(label, res['f1'], res['accuracy']))
        return res


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
    X_small = X_full[:1000]
    y_small = y_full[:1000]
    X_test = X_full[-500:]
    y_test = y_full[-500:]

    # X_small = [[n/255 for n in x] for x in X_small]
    # X_test = [[n/255 for n in x] for x in X_test]

    print("Testing plain data..")
    Run(X_small, y_small, X_test, y_test)

    print("Reducing dimensionality..")
    pca = PCA(n_components=800) #New dimentionality of feature vectors
    X_small_trans = pca.fit_transform(X_small)
    X_test_trans = pca.transform(X_test)

    print("Testing PCA data..")
    Run(X_test_trans, y_test, X_small_trans, y_small)


if __name__ == "__main__":
    main()