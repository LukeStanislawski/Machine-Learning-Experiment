import sys, os, time, datetime, logging, json, traceback
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
from scipy import stats

log = logging.getLogger("file_out")
hdlr = logging.FileHandler('task2.log')
formatter = logging.Formatter('"%(asctime)s [%(levelname)-5.5s]  %(message)s"')
hdlr.setFormatter(formatter)
log.addHandler(hdlr) 
log.setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

class TestCase():
    def __init__(self, 
    	X_train, 
    	y_train, 
    	X_test, 
    	y_test, 
    	folds=10, 
    	kernel='linear', 
    	C=1.0, 
    	degree=3, 
    	gamma='scale', 
    	ID="0", 
    	pca=0.0):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.folds = folds
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.ID = ID
        self.pca = pca

        self.res = {}
        self.res["test"] = {}
        self.res["train"] = []
        self.res["param"] = {}
        self.res["run"] = {}
        self.res["run"]["datetime"] = datetime.datetime.now().strftime("%x %X")
        self.res["param"]["folds"] = self.folds
        self.res["param"]["C"] = self.C
        self.res["param"]["kernel"] = self.kernel
        self.res["param"]["degree"] = self.degree
        self.res["param"]["gamma"] = self.gamma
        self.res["param"]["ID"] = self.ID
        self.res["param"]["pca"] = self.pca
        self.res["param"]["n_train"] = len(self.X_train)
        self.res["param"]["n_test"] = len(self.X_test)


        print("\n")
        log.info("Kernel: {}, C: {}, Folds: {}, PCA: {}, Degree: {}, Gamma: {}".format(self.kernel, self.C, self.folds, self.pca, self.degree, self.gamma))

        try:
        	self.svc = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, degree=self.degree)
        except Exception as e:
        	log.info("ERROR: Could not create SVC")
            # log.info(e)

        try:
        	self.run()
        except Exception as e:
            log.info("ERROR: while running test")
            log.info(str(e))
            log.info(sys.exc_info())
            log.info(traceback.format_exc())


    def run(self):
        tstart = time.time()

        self.apply_pca()
        tr_start = time.time()
        self.train()
        self.res["run"]["traintime"] = time.time() - tr_start
        self.res["test"] = self.test(self.X_test, self.y_test)
        
        self.res["run"]["runtime"] = time.time() - tstart
        self.log_results()


    def apply_pca(self):
        if float(self.pca) != 1.0:
            tstart = time.time()

            max_pca = min(len(self.X_train), len(self.X_train[0]))
            n = int(self.pca * max_pca)

            pca = PCA(n_components=n) #New dimentionality of feature vectors
            self.X_train = pca.fit_transform(self.X_train)
            self.X_test = pca.transform(self.X_test)

            self.res["run"]["pca_time"] = time.time() - tstart


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
            
            self.res["train"].append(it_res)
            log.info("{0} - f1: {1:.3f}, accuracy: {2:.3f}, train_time: {3:.3f}, test_time: {4:.3f}".format(i, it_res['f1'], it_res['accuracy'], it_res["t_train"], it_res["runtime"]))

        self.res["run"]["cv_f1"] = np.mean([x["f1"] for x in self.res["train"]])
        self.res["run"]["cv_f1_classes"] = [np.mean([x["f1_classes"][j] for x in self.res["train"]]) for j in len(list(set(self.y_train))) ]
        self.res["run"]["cv_accuracy"] = np.mean([x["accuracy"] for x in self.res["train"]])
        log.info("CV - f1: {:.3f}, accuracy: {:.3f}".format(self.res["run"]["cv_f1"], self.res["run"]["cv_accuracy"]))


    def test(self, X, y):
        tstart = time.time()
        y_pred = self.svc.predict(X)
        t_test = time.time() - tstart

        test_res={}
        test_res["n_test"] = len(X)
        test_res["runtime"] = t_test
        test_res['f1'] = f1_score(y, y_pred, average='micro')
        test_res['f1_classes'] = f1_score(y, y_pred, average=None)
        test_res['accuracy'] = accuracy_score(y, y_pred)
        return test_res


    def log_results(self):
        with open('results1.csv', 'a+') as f:
            f.write(json.dumps(self.res) + "\n")

        print("F - f1: {0:.3f}, accuracy: {1:.3f}, runtime: {2:.3f}".format(self.res["test"]["f1"], self.res["test"]["accuracy"], self.res["run"]["runtime"]))



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


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
    X_small = X_full[:3000]
    y_small = y_full[:3000]
    X_test = X_full[-300:]
    y_test = y_full[-300:]

    # X_small = [[n/255 for n in x] for x in X_small]
    # X_test = [[n/255 for n in x] for x in X_test]


    # Task 2.1
    id = 0
    for pca in [1.0, 0.7, 0.5, 0.3, 0.9, 0.8, 0.6, 0.4, 0.2]:
    	TestCase(X_small, y_small, X_test, y_test, ID=id, kernel='linear', pca=pca, C=1)
    	id += 1

    # Task 2.2
    for C in [1000, 500, 100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:
        for degree in [2, 4, 6]:
            TestCase(X_small, y_small, X_test, y_test, ID=id, kernel='poly', pca=1.0, C=C, degree=degree, , gamma='auto')
            id += 1
        TestCase(X_small, y_small, X_test, y_test, ID=id, kernel='rbf', pca=1.0, C=C, gamma='auto')
        id += 1


    for degree in range(9):
        for C in [3, 5, 7]:
            TestCase(X_small, y_small, X_test, y_test, ID=id, kernel='poly', pca=1.0, C=C, gamma='auto', degree=degree)
            id += 1


if __name__ == "__main__":
    main()



# def display_img(img):
#     img2 = [[img[x], img[1024+x], img[2048+x]] for x in range(int(len(img)/3))]  
#     img3 = [img2[x*32:(x+1)*32] for x in range(32)]
#     plt.imshow(img3)
#     plt.show()