import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
from utils import get_classes, get_label

rows = None
cols = None
gn = 0


def main(PATH):
	global rows
	global cols
	
	results = []
	with open(PATH) as f:
		lines = f.readlines()
	for li, line in enumerate(lines):
		results.append(json.loads(line))


	plt.figure(figsize=(15,8))
	plt.suptitle("Runtime: {:.3f}, Train Set: {}, Test Set: {}".format(
											sum([x["run"]["runtime"] for x in results]),
											results[0]["param"]["n_train"],
											results[0]["param"]["n_test"]))
	plt.subplots_adjust(bottom=0.08, top=0.9, left=0.08, right=0.95, wspace=0.2, hspace=0.4)
	rows = 3
	cols = 3

	#--- Task 2.1 - 2.2 ---
	acc_v_pca(results)
	f1_pc_v_dimensionality(results)
	# f1_v_pca(results)
	#--- Task 2.3 - 2.4 ---
	poly_c_v_acc(results)
	poly_c_v_f1(results)
	rbf_c_v_acc(results)
	rbf_c_v_f1(results)
	poly_degree_v_acc(results)
	poly_degree_v_f1(results)


	plt.show()
	# save_plots(results)	# For rendering in Markdown



# Graphing Functions
# ------------------

def f1_v_pca(results, sp=True):
	# f1 against PCA
	k_linear = [x for x in results if x["param"]["kernel"] == 'linear']
	k_linear = sorted(k_linear, key = lambda i: i["param"]['pca'])
	lin_pcas = [x["param"]["pca"] for x in k_linear]
	lin_f1 = [x["run"]["cv_f1"] for x in k_linear]

	if sp: plt.subplot(rows,cols,get_gn())
	plt.title('Linear Kernel F1 Score Against PCA Dimentionality')
	plt.ylabel('F1 Score')
	plt.xlabel('Proportion of Original Dimentionality')
	plt.plot(lin_pcas, lin_f1, get_lc(0))


def acc_v_pca(results, sp=True):
	if sp: plt.subplot(rows,cols,get_gn())
	plt.title('Linear Kernel Accuracy Against PCA Dimentionality')
	plt.ylabel('Accuracy')
	plt.xlabel('Proportion of Original Dimentionality')

	k_linear = [x for x in results if x["param"]["kernel"] == 'linear']
	k_linear = sorted(k_linear, key = lambda i: i["param"]['pca'])
	lin_pcas = [x["param"]["pca"] for x in k_linear]

	lin_acc = [x["test"]["accuracy"] for x in k_linear]
	plot(lin_pcas, lin_acc, i=1, label="Test Data")
	
	lin_cv_acc = [x["run"]["cv_accuracy"] for x in k_linear]
	plot(lin_pcas, lin_cv_acc, i=2, label="Cross Validation")

	plt.legend(loc="upper right")


def poly_c_v_acc(results, sp=True):
	# poly: C vs accuracy
	if sp: plt.subplot(rows,cols,get_gn())
	plt.title('Accuracy of Polynomial Kernel against C Value')
	plt.ylabel('Accuracy')
	plt.xlabel('log(C)')

	k_poly = [x for x in results if str(x["param"]["ID"]).startswith("PvC")]

	for degree in range(9):
		poly_vs_C = [x for x in k_poly if x["param"]["gamma"] == "scale" and x["param"]["degree"] == degree]
		if len(poly_vs_C) > 0:
			poly_vs_C = sorted(poly_vs_C, key = lambda i: i["param"]['C'])
			poly_vs_C_C = [np.log10(x["param"]["C"]) for x in poly_vs_C]
			poly_vs_C_acc = [x["run"]["cv_accuracy"] for x in poly_vs_C]
			plt.plot(poly_vs_C_C, poly_vs_C_acc, get_lc(degree), label="Degree={}".format(degree))

	plt.legend(loc="lower right")
	# plt.ylim(0.05, 0.3)


def poly_c_v_f1(results, sp=True):
	# poly: C vs f1
	degree = 3
	if sp: plt.subplot(rows,cols,get_gn())

	k_poly = [x for x in results if str(x["param"]["ID"]).startswith("PvC") and x["param"]["degree"] == degree]
	k_poly = sorted(k_poly, key = lambda i: i["param"]['C'])
	log_C = [np.log10(x["param"]["C"]) for x in k_poly]

	# scores = []
	for ci, c in enumerate(get_classes()):
		f1s = [x["test"]["f1_pc"][ci] for x in k_poly]
		plot(log_C, f1s, ci, label="{}".format(get_label(ci)))

	plt.title('Polynomial Kernel F1 Score Against C Value')
	plt.ylabel('F1')
	plt.xlabel('log(C)')
	plt.legend(loc="lower right")
	# plt.ylim(0.05, 0.3)


def rbf_c_v_acc(results, sp=True):
	# rbf: C vs accuracy
	if sp: plt.subplot(rows,cols,get_gn())
	plt.title('RBF Kernel Accuracy Against C Value')
	plt.ylabel('Accuracy')
	plt.xlabel('log(C)')

	k_rbf = [x for x in results if str(x["param"]["ID"]).startswith("RBFvC")]
	k_rbf = sorted(k_rbf, key = lambda i: i["param"]['C'])
	Cs = [np.log10(x["param"]["C"]) for x in k_rbf]
	
	t_accs = [x["test"]["accuracy"] for x in k_rbf]
	plot(Cs, t_accs, 0, label="Test")
	cv_accs = [x["run"]["cv_accuracy"] for x in k_rbf]
	plot(Cs, cv_accs, 1, label="CV")
	plt.legend(loc="lower right")
	# plt.ylim(0.05, 0.3)


def rbf_c_v_f1(results, sp=True):
	# rbf: C vs f1
	if sp: plt.subplot(rows,cols,get_gn())
	plt.title('RBF Kernel F1 Score Against C Value')
	plt.ylabel('F1')
	plt.xlabel('log(C)')

	k_rbf = [x for x in results if str(x["param"]["ID"]).startswith("RBFvC")]
	k_rbf = sorted(k_rbf, key = lambda i: i["param"]['C'])
	Cs = [np.log10(x["param"]["C"]) for x in k_rbf]

	for ci, c in enumerate(get_classes()):
		f1s = [x["test"]["f1_pc"][ci] for x in k_rbf]
		plt.plot(Cs, f1s, get_lc(3))
		plot(Cs, f1s, i=ci, label=get_label(ci))
	
	
	plt.legend(loc="lower right")
	# f1s = [x["run"]["cv_f1"] for x in k_rbf]
	# plt.ylim(0.05, 0.3)


def poly_degree_v_acc(results, sp=True):
	if sp: plt.subplot(rows,cols,get_gn())

	k_poly = [x for x in results if x["param"]["kernel"] == 'poly']
	# Cs = list(set([x["param"]["C"] for x in k_poly]))
	# Cs = [2,3,4]
	Cs = [3]
	for i, C in enumerate(Cs):
		poly_vs_deg = [x for x in k_poly if str(x["param"]["ID"]).startswith("PvD") and x["param"]["C"] == C]
		poly_vs_deg = sorted(poly_vs_deg, key = lambda i: i["param"]['degree'])
		poly_vs_deg_deg = [x["param"]["degree"] for x in poly_vs_deg]
		poly_vs_deg_cv_acc = [x["run"]["cv_accuracy"] for x in poly_vs_deg]
		poly_vs_deg_test_acc = [x["test"]["accuracy"] for x in poly_vs_deg]
		plt.plot(poly_vs_deg_deg, poly_vs_deg_cv_acc, get_lc(0), label="CV Accuracy")
		plt.plot(poly_vs_deg_deg, poly_vs_deg_test_acc, get_lc(1), label="Test Accuracy")

	plt.title('Polynomial Kernel Accuracy Against Degree')
	plt.ylabel('Accuracy')
	plt.xlabel('Degree')
	plt.legend(loc="upper right")
	# plt.ylim(0.05, 0.3)


def poly_degree_v_f1(results, sp=True):
	if sp: plt.subplot(rows,cols,get_gn())

	k_poly = [x for x in results if x["param"]["kernel"] == 'poly']
	# Cs = list(set([x["param"]["C"] for x in k_poly]))
	Cs = [2,3,4]
	for i, C in enumerate(Cs):
		poly_vs_deg = [x for x in k_poly if str(x["param"]["ID"]).startswith("PvD") and x["param"]["C"] == C]
		poly_vs_deg = sorted(poly_vs_deg, key = lambda i: i["param"]['degree'])
		poly_vs_deg_deg = [x["param"]["degree"] for x in poly_vs_deg]
		poly_vs_deg_f1 = [x["run"]["cv_f1"] for x in poly_vs_deg]
		plt.plot(poly_vs_deg_deg, poly_vs_deg_f1, get_lc(i), label="C={}".format(C))

	plt.title('Polynomial Kernel F1 Score Against Degree')
	plt.ylabel('F1')
	plt.xlabel('Degree')
	plt.legend(loc="upper right")
	# plt.ylim(0.05, 0.3)



def f1_pc_v_dimensionality(results, sp=True):
	if sp: plt.subplot(rows,cols,get_gn())

	k_linear = [x for x in results if x["param"]["kernel"] == 'linear']
	k_linear = sorted(k_linear, key = lambda i: i["param"]['pca'])
	pcas = [x["param"]["pca"] for x in k_linear]
	
	# class_scores = []
	for ci, c in enumerate(get_classes()):
		res = {}
		res["f1s"] = [x["test"]["f1_pc"][ci] for x in k_linear]
		res["precision_pcs"] = [x["test"]["precision_pc"][ci] for x in k_linear]
		res["recall_pcs"] = [x["test"]["recall_pc"][ci] for x in k_linear]

		plt.plot(pcas, res["f1s"], label=get_label(ci), marker=get_m(), linestyle=get_ls(),
          markerfacecolor=get_c(ci), markeredgecolor=get_c(ci), color=get_c(ci))

	plt.title('F1 Score Against Proportion of Dimentionality')
	plt.ylabel('F1 Score')
	plt.xlabel('Proportion of Dimentionality')
	plt.legend(loc="upper right")



# Plotting helper functions
# -------------------------

def plot(x, y, i=0, label=None):
	plt.plot(x, y, 
		label=label, 
		marker=get_m(), 
		linestyle=get_ls(),
        markerfacecolor=get_c(i), 
        markeredgecolor=get_c(i), 
        color=get_c(i))

# Graph number incrementer
def get_gn():
	global gn
	gn += 1
	return gn

# Graph point code
def get_m():
	return "o"

# Graph line type
def get_ls():
	return "--"

# Color codes
def get_c(index):
	cs = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#333399', '#BDC3C7', '#34495E']
	return cs[index]

# Matplotlib line code
# DEPRECATED
def get_lc(index):
	cs = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'r', 'g', 'r']
	lcs = ["{}{}{}".format(get_m(), c, get_ls()) for c in cs]
	return lcs[index]


# Writing plots to file
# ---------------------

def save_plots(results):
	p = "figs/{}.png"

	plt.clf()
	plt.figure(figsize=(10,6))
	
	f1_v_pca(results, sp=False)
	plt.savefig(p.format("f1_v_pca"))

	plt.clf()
	acc_v_pca(results, sp=False)
	plt.savefig(p.format("acc_v_pca"))

	plt.clf()
	f1_pc_v_dimensionality(results, sp=False)
	plt.savefig(p.format("f1_pc_v_dimensionality"))

	plt.clf()
	poly_c_v_acc(results, sp=False)
	plt.savefig(p.format("poly_c_v_acc"))

	plt.clf()
	poly_c_v_f1(results, sp=False)
	plt.savefig(p.format("poly_c_v_f1"))

	plt.clf()
	rbf_c_v_acc(results, sp=False)
	plt.savefig(p.format("rbf_c_v_acc"))

	plt.clf()
	rbf_c_v_f1(results, sp=False)
	plt.savefig(p.format("rbf_c_v_f1"))

	plt.clf()
	poly_degree_v_acc(results, sp=False)
	plt.savefig(p.format("poly_degree_v_acc"))

	plt.clf()
	poly_degree_v_f1(results, sp=False)
	plt.savefig(p.format("poly_degree_v_f1"))

	print("Images saved")


# If no filepath passed in use results.csv
if __name__ == "__main__":
	if len(sys.argv) > 1:
		main(str(sys.argv[1]))
	else:
		f_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv')
		main(f_path)
		