import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
from utils import get_classes, get_label, ft, load_results

rows = None
cols = None
gn = 0


def main():
	results = load_results(os.path.abspath(__file__))

	plt.figure(figsize=(15,8))
	plt.suptitle("X_Train: {}, X_Train: {}, Tests Run: {}, Runtime: {}, Max: {}, Min: {}".format(
				results[0]["param"]["n_train"],
				results[0]["param"]["n_test"],
				len(results),
				ft(sum([x["run"]["runtime"] for x in results])),
				ft(max([x["run"]["runtime"] for x in results])),
				ft(min([x["run"]["runtime"] for x in results]))))
	plt.subplots_adjust(bottom=0.08, top=0.9, left=0.08, right=0.95, wspace=0.2, hspace=0.4)
	
	# Rows and cols to display
	# Note: All graphs are written to file when save_plots() is called
	# regardless of whether they are called here in main()
	global rows
	global cols
	rows = 4
	cols = 3

	# Comment out any graphs you wish to not display and adjust vals of rows & cols
	acc_v_pca(results)
	f1_pc_v_dimensionality(results)
	f1_v_pca(results)
	poly_c_v_acc(results)
	poly_c_v_f1(results)
	poly_degree_v_acc(results)
	poly_degree_v_f1(results)
	rbf_c_v_acc(results)
	rbf_c_v_f1(results)
	rbf_gamma_v_acc(results)
	rbf_gamma_v_f1(results)

	# Displayes graphs on screen
	plt.show()

	# Saves graphs to file for rendering in report
	# save_plots(results)



# Graphing Functions
# ------------------

def f1_v_pca(results, sp=True):
	# f1 against PCA
	k_linear = [x for x in results if x["param"]["kernel"] == 'linear']
	k_linear = sorted(k_linear, key = lambda i: i["param"]['pca'])
	lin_pcas = [x["param"]["pca"] for x in k_linear]
	lin_f1 = [x["run"]["cv_f1"] for x in k_linear]

	if sp: plt.subplot(rows,cols,get_gn())
	plt.title('Linear Kernel: F1 Score Against PCA Dimentionality')
	plt.ylabel('F1 Score')
	plt.xlabel('Proportion of Original Dimentionality')
	plt.plot(lin_pcas, lin_f1, get_lc(0))


def acc_v_pca(results, sp=True):
	if sp: plt.subplot(rows,cols,get_gn())
	plt.title('Linear Kernel: Accuracy Against PCA Dimentionality')
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
	plt.title('Polynomial Kernel: Accuracy Against C Value')
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


def poly_c_v_f1(results, sp=True):
	# poly: C vs f1
	if sp: plt.subplot(rows,cols,get_gn())
	k_poly = [x for x in results if str(x["param"]["ID"]).startswith("PvC") and x["param"]["degree"] == 3]
	k_poly = sorted(k_poly, key = lambda i: i["param"]['C'])
	log_C = [np.log10(x["param"]["C"]) for x in k_poly]

	for ci, c in enumerate(get_classes()):
		f1s = [x["test"]["f1_pc"][ci] for x in k_poly]
		plot(log_C, f1s, ci, label="{}".format(get_label(ci)))

	plt.title('Polynomial Kernel: F1 Score Against C Value')
	plt.ylabel('F1')
	plt.xlabel('log(C)')
	plt.legend(loc="lower right")


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
	plot(Cs, t_accs, 0, label="Test Accuracy")
	cv_accs = [x["run"]["cv_accuracy"] for x in k_rbf]
	plot(Cs, cv_accs, 1, label="CV Accuracy")
	t_f1s = [x["test"]["f1"] for x in k_rbf]
	plot(Cs, t_f1s, 2, label="Test F1")
	cv_f1s = [x["run"]["cv_f1"] for x in k_rbf]
	plot(Cs, cv_f1s, 3, label="CV F1")

	plt.legend(loc="lower right")


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


def poly_degree_v_acc(results, sp=True):
	if sp: plt.subplot(rows,cols,get_gn())

	k_poly = [x for x in results if x["param"]["kernel"] == 'poly']
	
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


def poly_degree_v_f1(results, sp=True):
	if sp: plt.subplot(rows,cols,get_gn())

	k_poly = [x for x in results if x["param"]["kernel"] == 'poly']
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



def f1_pc_v_dimensionality(results, sp=True):
	if sp: plt.subplot(rows,cols,get_gn())

	k_linear = [x for x in results if x["param"]["kernel"] == 'linear']
	k_linear = sorted(k_linear, key = lambda i: i["param"]['pca'])
	pcas = [x["param"]["pca"] for x in k_linear]
	
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



def rbf_gamma_v_acc(results, sp=True):
	# rbf: C vs accuracy
	if sp: plt.subplot(rows,cols,get_gn())
	plt.title('RBF Kernel Accuracy Against Gamma Value')
	plt.ylabel('Accuracy')
	plt.xlabel('log(gamma)')

	k_rbf = [x for x in results if str(x["param"]["ID"]).startswith("RBFvgamma")]
	k_rbf = sorted(k_rbf, key = lambda i: i["param"]['gamma'])
	Cs = [np.log10(x["param"]["gamma"]) for x in k_rbf]
	
	t_accs = [x["test"]["accuracy"] for x in k_rbf]
	plot(Cs, t_accs, 0, label="Test")
	cv_accs = [x["run"]["cv_accuracy"] for x in k_rbf]
	plot(Cs, cv_accs, 1, label="CV")
	plt.legend(loc="lower right")
	# plt.ylim(0.05, 0.3)



def rbf_gamma_v_f1(results, sp=True):
	# rbf: C vs f1
	if sp: plt.subplot(rows,cols,get_gn())
	plt.title('RBF Kernel F1 Score Against Gamma Value')
	plt.ylabel('F1')
	plt.xlabel('log(gamma)')

	k_rbf = [x for x in results if str(x["param"]["ID"]).startswith("RBFvgamma")]
	k_rbf = sorted(k_rbf, key = lambda i: i["param"]['gamma'])
	Cs = [np.log10(x["param"]["gamma"]) for x in k_rbf]

	for ci, c in enumerate(get_classes()):
		f1s = [x["test"]["f1_pc"][ci] for x in k_rbf]
		plt.plot(Cs, f1s, get_lc(3))
		plot(Cs, f1s, i=ci, label=get_label(ci))
	
	
	plt.legend(loc="lower right")


# Plotting helper functions
# -------------------------
# Plot line
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
def get_m(): return "o"

# Graph line type
def get_ls(): return "--"

# Color codes
def get_c(index):
	cs = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#333399', '#BDC3C7', '#34495E']
	return cs[index]

# Matplotlib line code
# -- DEPRECATED --
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

	plt.clf()
	rbf_gamma_v_acc(results, sp=False)
	plt.savefig(p.format("rbf_gamma_v_acc"))

	plt.clf()
	rbf_gamma_v_f1(results, sp=False)
	plt.savefig(p.format("rbf_gamma_v_f1"))

	print("Images saved to figs/")


if __name__ == "__main__":
	main()
		