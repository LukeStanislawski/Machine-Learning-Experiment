import sys, os, json
import matplotlib.pyplot as plt
import numpy as np

rows = None
cols = None
gn = 0


def main():
	global rows
	global cols
	results = []
	with open("results1.csv") as f:
		lines = f.readlines()

	for li, line in enumerate(lines):
		results.append(json.loads(line))


	plt.figure(figsize=(15,8))
	plt.subplots_adjust(bottom=0.08, top=0.9, left=0.08, right=0.95, wspace=0.2, hspace=0.4)
	rows = 3
	cols = 3

	f1_v_pca(results)
	acc_v_pca(results)
	poly_c_v_acc(results)
	poly_c_v_f1(results)
	rbf_c_v_acc(results)
	rbf_c_v_f1(results)
	ploy_degree_v_acc(results)
	ploy_degree_v_f1(results)


	# General stats
	t_runtime = sum([x["run"]["runtime"] for x in results])
	print("Total runtime was {:.3f} seconds".format(t_runtime))

	plt.show()




def f1_v_pca(results):
	# f1 against PCA
	k_linear = [x for x in results if x["param"]["kernel"] == 'linear']
	k_linear = sorted(k_linear, key = lambda i: i["param"]['pca'])
	lin_pcas = [x["param"]["pca"] for x in k_linear]
	lin_f1 = [x["run"]["cv_f1"] for x in k_linear]

	plt.subplot(rows,cols,get_gn())
	plt.title('F1 Against PCA Dimentionality')
	plt.ylabel('F1 Score')
	plt.xlabel('Proportion of Original Dimentionality')
	plt.plot(lin_pcas, lin_f1, 'bs-')


def acc_v_pca(results):	
	k_linear = [x for x in results if x["param"]["kernel"] == 'linear']
	k_linear = sorted(k_linear, key = lambda i: i["param"]['pca'])
	lin_pcas = [x["param"]["pca"] for x in k_linear]
	lin_acc = [x["run"]["cv_accuracy"] for x in k_linear]

	plt.subplot(rows,cols,get_gn())
	plt.title('Accuracy Against PCA Dimentionality')
	plt.ylabel('Accuracy')
	plt.xlabel('Proportion of Original Dimentionality')
	plt.plot(lin_pcas, lin_acc, 'gs-')


def poly_c_v_acc(results):
	# poly: C vs accuracy
	plt.subplot(rows,cols,get_gn())
	graph_codes = ["gs-", "rs--", "bs--"]

	k_poly = [x for x in results if x["param"]["kernel"] == 'poly']
	# degrees = list(set([x["param"]["degree"] for x in k_poly]))
	degrees = [2,4,6]
	for i, degree in enumerate(degrees):
		poly_vs_C = [x for x in k_poly if x["param"]["gamma"] == "scale" and x["param"]["degree"] == degree]
		poly_vs_C = sorted(poly_vs_C, key = lambda i: i["param"]['C'])
		poly_vs_C_C = [np.log10(x["param"]["C"]) for x in poly_vs_C]
		poly_vs_C_acc = [x["run"]["cv_accuracy"] for x in poly_vs_C]
		plt.plot(poly_vs_C_C, poly_vs_C_acc, graph_codes[i], label="Degree={}".format(degree))


	plt.title('Poly Accuracy of C Value')
	plt.ylabel('Accuracy')
	plt.xlabel('log(C)')
	plt.legend(loc="lower right")
	# plt.ylim(0.05, 0.3)


def poly_c_v_f1(results):
	# poly: C vs f1
	plt.subplot(rows,cols,get_gn())
	graph_codes = ["gs-", "rs--", "bs--"]

	k_poly = [x for x in results if x["param"]["kernel"] == 'poly']
	# degrees = list(set([x["param"]["degree"] for x in k_poly]))
	degrees = [2,4,6]
	for i, degree in enumerate(degrees):
		poly_vs_C = [x for x in k_poly if x["param"]["gamma"] == "scale" and x["param"]["degree"] == degree]
		poly_vs_C = sorted(poly_vs_C, key = lambda i: i["param"]['C'])
		poly_vs_C_C = [np.log10(x["param"]["C"]) for x in poly_vs_C]
		poly_vs_C_f1 = [x["run"]["cv_f1"] for x in poly_vs_C]
		plt.plot(poly_vs_C_C, poly_vs_C_f1, graph_codes[i], label="Degree={}".format(degree))

	plt.title('Poly F1 Against C Value')
	plt.ylabel('F1')
	plt.xlabel('log(C)')
	plt.legend(loc="lower right")
	# plt.ylim(0.05, 0.3)


def rbf_c_v_acc(results):
	# rbf: C vs accuracy
	plt.subplot(rows,cols,get_gn())
	k_rbf = [x for x in results if x["param"]["kernel"] == 'rbf' and x["param"]["gamma"] == "auto"]
	k_rbf = sorted(k_rbf, key = lambda i: i["param"]['C'])
	
	Cs = [np.log10(x["param"]["C"]) for x in k_rbf]
	accs = [x["run"]["cv_accuracy"] for x in k_rbf]
	plt.plot(Cs, accs, "gs-")
	plt.title('RBF Accuracy Against C Value')
	plt.ylabel('Accuracy')
	plt.xlabel('log(C)')
	# plt.ylim(0.05, 0.3)


def rbf_c_v_f1(results):
	# rbf: C vs f1
	plt.subplot(rows,cols,get_gn())
	k_rbf = [x for x in results if x["param"]["kernel"] == 'rbf' and x["param"]["gamma"] == "auto"]
	k_rbf = sorted(k_rbf, key = lambda i: i["param"]['C'])
	
	Cs = [np.log10(x["param"]["C"]) for x in k_rbf]
	f1s = [x["run"]["cv_f1"] for x in k_rbf]
	plt.plot(Cs, f1s, "gs-")
	plt.title('RBF F1 Against C Value')
	plt.ylabel('F1')
	plt.xlabel('log(C)')
	# plt.ylim(0.05, 0.3)


def ploy_degree_v_acc(results):
	plt.subplot(rows,cols,get_gn())
	graph_codes = ["gs-", "rs--", "bs--"]

	k_poly = [x for x in results if x["param"]["kernel"] == 'poly']
	# Cs = list(set([x["param"]["C"] for x in k_poly]))
	Cs = [3,5,7]
	for i, C in enumerate(Cs):
		poly_vs_deg = [x for x in k_poly if x["param"]["gamma"] == "auto" and x["param"]["C"] == C]
		poly_vs_deg = sorted(poly_vs_deg, key = lambda i: i["param"]['degree'])
		poly_vs_deg_deg = [x["param"]["degree"] for x in poly_vs_deg]
		poly_vs_deg_acc = [x["run"]["cv_accuracy"] for x in poly_vs_deg]
		plt.plot(poly_vs_deg_deg, poly_vs_deg_acc, graph_codes[i], label="C={}".format(C))

	plt.title('Poly Accuracy Against Degree')
	plt.ylabel('Accuracy')
	plt.xlabel('Degree')
	plt.legend(loc="upper right")
	# plt.ylim(0.05, 0.3)


def ploy_degree_v_f1(results):
	plt.subplot(rows,cols,get_gn())
	graph_codes = ["gs-", "rs--", "bs--", "ys--", "os--", "ps--"]

	k_poly = [x for x in results if x["param"]["kernel"] == 'poly']
	# Cs = list(set([x["param"]["C"] for x in k_poly]))
	Cs = [3,5,7]
	for i, C in enumerate(Cs):
		poly_vs_deg = [x for x in k_poly if x["param"]["gamma"] == 'auto' and x["param"]["C"] == C]
		poly_vs_deg = sorted(poly_vs_deg, key = lambda i: i["param"]['degree'])
		poly_vs_deg_deg = [x["param"]["degree"] for x in poly_vs_deg]
		poly_vs_deg_f1 = [x["run"]["cv_f1"] for x in poly_vs_deg]
		plt.plot(poly_vs_deg_deg, poly_vs_deg_f1, graph_codes[i], label="C={}".format(C))

	plt.title('Poly F1 Against Degree')
	plt.ylabel('F1')
	plt.xlabel('Degree')
	plt.legend(loc="upper right")
	# plt.ylim(0.05, 0.3)



def get_gn():
	global gn
	gn += 1
	return gn


if __name__ == "__main__":
	main()