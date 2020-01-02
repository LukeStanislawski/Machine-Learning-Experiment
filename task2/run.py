import sys, os, time, datetime, logging, json, traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from scipy import stats
from utils import get_logger, silence_warnings, ft

log = get_logger(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task2.log'))
f_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv')


class TestCase():
    """
    This class is used to run a test of an SVM. 
    The parameters can be passed in, or the SVC() itself. The class will
    log all results to results.csv in JSON. Each new line of JSON is a new 
    testcase.
    """
    def __init__(self, X_train, y_train, X_test, y_test, folds=10, kernel='linear', 
            C=1.0, degree=3, gamma='scale', ID="0", pca=0.0):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.classes = sorted(list(set(self.y_train)))  # = [0,1,2,3,4,5,6,7,8,9]
        self.folds = folds
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.ID = ID
        self.pca = pca

        # The results dict will get written to file as JSON
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
            log.info(str(e))
            log.info(sys.exc_info())
            log.info(traceback.format_exc())

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
        self.train()
        self.res["test"] = self.test(self.X_test, self.y_test)
        self.res["run"]["runtime"] = time.time() - tstart
        self.log_results()


    def apply_pca(self):
        # If self.pca = 1 then no PCA is performed, otherwise PCA is performed
        # to the specified proportion of train data set
        if float(self.pca) != 1.0:
            tstart = time.time()

            max_pca = min(len(self.X_train), len(self.X_train[0]))
            n = int(self.pca * max_pca)

            pca = PCA(n_components=n) #New dimentionality of feature vectors
            self.X_train = pca.fit_transform(self.X_train)
            self.X_test = pca.transform(self.X_test)

            self.res["run"]["pca_time"] = time.time() - tstart


    def train(self):
        """
        Cross validation is performed, saving results to self.res with each iteration
        """
        tr_start = time.time()
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
            log.info("{0} - f1: {1:.3f}, accuracy: {2:.3f}, train_time: {3}, test_time: {4}".format(i, it_res['f1'], it_res['accuracy'], ft(it_res["t_train"]), ft(it_res["runtime"])))

        self.calculate_results()
        self.res["run"]["traintime"] = time.time() - tr_start
        log.info("CV- f1: {:.3f}, accuracy: {:.3f}, runtime: {}".format(self.res["run"]["cv_f1"], self.res["run"]["cv_accuracy"], ft(self.res["run"]["traintime"])))


    def test(self, X, y):
        """
        Tests the self.svc on any passed in dataset
        Results are returned in dict form
        """
        tstart = time.time()
        y_pred = self.svc.predict(X)

        test_res={}
        test_res["n_test"] = len(X)
        test_res["runtime"] = time.time() - tstart
        test_res['f1'] = f1_score(y, y_pred, average='macro')
        test_res['f1_pc'] = list(f1_score(y, y_pred, labels=self.classes, average=None))
        test_res['accuracy'] = accuracy_score(y, y_pred)
        test_res['precision'] = precision_score(y, y_pred, average='macro')
        test_res['precision_pc'] = list(precision_score(y, y_pred, labels=self.classes, average=None))
        test_res['recall'] = recall_score(y, y_pred, average='macro')
        test_res['recall_pc'] = list(recall_score(y, y_pred, labels=self.classes, average=None))

        return test_res


    def calculate_results(self):
        """
        Calculates final results of cross validation
        Results saved to self.res dict
        """
        self.res["run"]["cv_f1"] = np.mean([x["f1"] for x in self.res["train"]])
        self.res["run"]["cv_accuracy"] = np.mean([x["accuracy"] for x in self.res["train"]])
        self.res["run"]["cv_precision"] = np.mean([x["precision"] for x in self.res["train"]])
        self.res["run"]["cv_recall"] = np.mean([x["recall"] for x in self.res["train"]])
        
        self.res["run"]["cv_f1_pc"] = []
        self.res["run"]["cv_precision_pc"] = []
        self.res["run"]["cv_recall_pc"] = []
        
        for ci, c in enumerate(self.classes):
            f1s = prs = rcs = []
            for t in self.res["train"]:
                f1s.append(t["f1_pc"][ci])
                prs.append(t["precision_pc"][ci])
                rcs.append(t["recall_pc"][ci])

            self.res["run"]["cv_f1_pc"].append(np.mean(f1s))
            self.res["run"]["cv_precision_pc"].append(np.mean(prs))
            self.res["run"]["cv_recall_pc"].append(np.mean(rcs))


    def log_results(self):
        """
        Results dict written to file as JSON
        JSON is written on a single line and appended to results file
        """
        with open(f_path, 'a+') as f:
            f.write(json.dumps(self.res) + "\n")

        print("T - f1: {0:.3f}, accuracy: {1:.3f}, runtime: {2}".format(self.res["test"]["f1"], self.res["test"]["accuracy"], ft(self.res["run"]["runtime"])))



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
        data = unpickle("../cifar-10-batches-py/{}".format(batch_label))
        X_full.extend(data[b"data"])
        y_full.extend(data[b"labels"])

    return X_full, y_full



def main():
    print("Loading data")
    n_train = 4000
    n_test = 1000
    X_full, y_full = load_data()
    X_small = X_full[:n_train]
    y_small = y_full[:n_train]
    X_test = X_full[-1*n_test:]
    y_test = y_full[-1*n_test:]

    print("Normalising")
    X_small = [[n/255 for n in x] for x in X_small]
    X_test = [[n/255 for n in x] for x in X_test]


    # Task 2.1
    # with open(f_path) as f: id = len(f.readlines())
    id = 0

    # Test different levels of dimensionality
    for pca in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
    	TestCase(X_small, y_small, X_test, y_test, ID=id, kernel='linear', pca=pca, C=1)
    	id += 1


    # # Test poly kernel with different values of C
    # for C_e in range(-5,6):
    #     for degree in [2,3,4]:
    #         TestCase(X_small, y_small, X_test, y_test, ID="PvC{}".format(id), kernel='poly', pca=1.0, C=10 ** C_e, degree=degree, gamma='scale')
    #         id += 1


    # # Test Poly kernel with different degrees
    # for degree in range(12):
    #     TestCase(X_small, y_small, X_test, y_test, ID="PvD{}".format(id), kernel='poly', pca=1.0, C=3, gamma='scale', degree=degree)
    #     id += 1


    # Test RBF kernel with different C parameters
    # for C_e in [-2, -1, 0, 1, 2, 3, 4]:
    # for C_e in [-3, 3, 4]:
    #     TestCase(X_small, y_small, X_test, y_test, ID="RBFvC{}".format(id), kernel='rbf', pca=1.0, C=10 ** C_e, gamma='scale')
    #     id += 1


    # # Test RBF kernel with different gamma parameter
    # for gamma_e in [0, -2, -4, -6, -8, -10]:
    #     TestCase(X_small, y_small, X_test, y_test, ID="RBFvgamma{}".format(id), kernel='rbf', pca=1.0, C=3, gamma=10 ** gamma_e)
    #     id += 1


if __name__ == "__main__":
    silence_warnings()
    main()
