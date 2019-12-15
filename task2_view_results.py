import sys, os
import matplotlib.pyplot as plt

with open("results1.csv") as f:
	lines = f.readlines()

plt.figure(figsize=(15,5))
rows = 1
cols = 2

results = []
labels = ["ID","datetime","kernel","C","degree","gamma","folds","pca","f1","accuracy","X_train,","X_test,","runtime","pca_time","train_time","runtime"]

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

plt.show()