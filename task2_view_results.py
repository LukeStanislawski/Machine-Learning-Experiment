import sys, os
import matplotlib.pyplot as plt
import numpy as np

with open("results1.csv") as f:
	lines = f.readlines()

plt.figure(figsize=(15,8))
rows = 2
cols = 4

results = []
labels = ["ID","datetime","kernel","C","degree","gamma","folds","pca","f1","accuracy","X_train,","X_test,","runtime","pca_time","train_time","test_runtime"]

for li, line in enumerate(lines):
	if len(line) > 0 and li != 0:
		data = line.split(",")
		res = {}
		for i, d in enumerate(data):
			res[labels[i]] = d

		results.append(res)


# Accuracy/f1 against PCA
k_linear = [x for x in results if x["kernel"] == 'linear']
for i, x in enumerate(k_linear):
	if x['pca'] == "None":
		k_linear[i]['pca'] = "1.0"

k_linear = sorted(k_linear, key = lambda i: i['pca'])
lin_pcas = [float(x["pca"]) for x in k_linear]
lin_acc = [float(x["accuracy"]) for x in k_linear]
lin_f1 = [float(x["f1"]) for x in k_linear]

plt.subplot(rows,cols,1)
plt.title('F1 Against PCA Dimentionality')
plt.ylabel('F1 Score')
plt.xlabel('Proportion of Original Dimentionality')
plt.plot(lin_pcas, lin_f1, 'bs-')

plt.subplot(rows,cols,2)
plt.title('Accuracy Against PCA Dimentionality')
plt.ylabel('Accuracy')
plt.xlabel('Proportion of Original Dimentionality')
plt.plot(lin_pcas, lin_acc, 'gs-')


# poly: C vs accuracy
plt.subplot(rows,cols,3)
graph_codes = ["gs-", "rs--", "bs--"]
k_poly = [x for x in results if x["kernel"] == 'poly']
for i, degree in enumerate(["2", "3", "4"]):
	poly_vs_C = [x for x in k_poly if x["gamma"] == "auto" and x["degree"] == degree]
	poly_vs_C = sorted(poly_vs_C, key = lambda i: i['C'])
	poly_vs_C_C = [np.log10(float(x["C"])) for x in poly_vs_C]
	poly_vs_C_acc = [float(x["accuracy"]) for x in poly_vs_C]
	plt.plot(poly_vs_C_C, poly_vs_C_acc, graph_codes[i], label="Degree={}".format(degree))

plt.title('Poly Accuracy of C value')
plt.ylabel('Accuracy')
plt.xlabel('log(C)')
plt.legend(loc="upper right")
plt.ylim(0.05, 0.3)


# poly: C vs f1
plt.subplot(rows,cols,4)
graph_codes = ["gs-", "rs--", "bs--"]
k_poly = [x for x in results if x["kernel"] == 'poly']
for i, degree in enumerate(["2", "3", "4"]):
	poly_vs_C = [x for x in k_poly if x["gamma"] == "auto" and x["degree"] == degree]
	poly_vs_C = sorted(poly_vs_C, key = lambda i: i['C'])
	poly_vs_C_C = [np.log10(float(x["C"])) for x in poly_vs_C]
	poly_vs_C_f1 = [float(x["f1"]) for x in poly_vs_C]
	plt.plot(poly_vs_C_C, poly_vs_C_f1, graph_codes[i], label="Degree={}".format(degree))

plt.title('Poly F1 Against C value')
plt.ylabel('F1')
plt.xlabel('log(C)')
plt.legend(loc="upper right")
plt.ylim(0.05, 0.3)


# rbf: C vs accuracy
plt.subplot(rows,cols,5)
k_rbf = [x for x in results if x["kernel"] == 'rbf']
rbf_vs_C = [x for x in k_rbf if x["gamma"] == "auto"]
rbf_vs_C = sorted(rbf_vs_C, key = lambda i: i['C'])
rbf_vs_C_C = [np.log10(float(x["C"])) for x in rbf_vs_C]
rbf_vs_C_acc = [float(x["accuracy"]) for x in rbf_vs_C]
plt.plot(rbf_vs_C_C, rbf_vs_C_acc, "gs-")
plt.title('RBF Accuracy Against C value')
plt.ylabel('Accuracy')
plt.xlabel('log(C)')
# plt.ylim(0.05, 0.3)


# General stats
t_runtime = sum([float(x["runtime"]) for x in results])
print("Total runtime was {} seconds.".format(t_runtime))

plt.show()