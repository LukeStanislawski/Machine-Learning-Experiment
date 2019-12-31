import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
from utils import get_classes, get_label, ft

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
	plt.suptitle("X_Train: {}, X_Train: {}, Tests Run: {}, Runtime: {}, Max: {}, Min: {}".format(
				results[0]["params"]["n_train"],
				results[0]["params"]["n_test"],
				len(results),
				ft(sum([x["runtime"] if "runtime" in x else 0.0 for x in results])),
				ft(max([x["runtime"] if "runtime" in x else 0.0 for x in results])),
				ft(min([x["runtime"] if "runtime" in x else 0.0 for x in results]))))
	plt.subplots_adjust(bottom=0.08, top=0.9, left=0.08, right=0.95, wspace=0.2, hspace=0.4)
	rows = 3
	cols = 3


	FCN_acc_v_hl(results)
	FCN_ttrain_v_hl(results)
	Conv_nc_v_acc(results)
	Pool_Comparison(results)
	Activation_Comparison_acc(results)
	Activation_Comparison_f1(results)
	channel_acc(results)
	epoch_acc(results)


	plt.show()
	save_plots(results)	# For rendering in Markdown



# Graphing Functions
# ------------------

def FCN_acc_v_hl(results, sp=True):
	if sp: plt.subplot(rows,cols,get_gn())
	plt.title('FCN Accuracy Against n Hidden Layers')
	plt.ylabel('Accuracy')
	plt.xlabel('n')

	fcns = [x for x in results if x["params"]["model"] == "FCN"]
	hl = [x["params"]["hidden_layers"] for x in fcns]
	accs = [x["test"]["accuracy"] for x in fcns]
	f1s = [np.mean(x["test"]["f1_pc"]) for x in fcns]
	rts = [ft(x["runtime"]) for x in fcns]
	plot(hl, accs, 0)

	if sp:
		print("\nFCN Accuracy Against n Hidden Layers")
		print(hl)
		print(accs)
		print(f1s)
		print(" ".join(rts))


def FCN_ttrain_v_hl(results, sp=True):
	if sp: plt.subplot(rows,cols,get_gn())
	plt.title('FCN Train time Against n Hidden Layers')
	plt.ylabel('Train time / seconds')
	plt.xlabel('n')

	fcns = [x for x in results if x["params"]["model"] == "FCN"]
	hl = [x["params"]["hidden_layers"] for x in fcns]
	times = [x["train"]["runtime"] for x in fcns]
	plot(hl, times, 0)


def Conv_nc_v_acc(results, sp=True):
	if sp: plt.subplot(rows,cols,get_gn())
	plt.title('ConvX Accuracy Against n Convolution & Max Pooling Layers')
	plt.ylabel('Accuracy')
	plt.xlabel('n Convolution & Max Pooling Layers')

	fcns = [x for x in results if x["params"]["model"] == "ConvX"]
	cl = [x["params"]["id"] for x in fcns]
	accs = [x["test"]["accuracy"] for x in fcns]
	f1s = [np.mean(x["test"]["f1_pc"]) for x in fcns]
	rts = [ft(x["runtime"]) for x in fcns]
	plot(cl, accs, 0)

	if sp:
		print("\nConvX Accuracy against N convolution layers")
		print(cl)
		print(accs)
		print(f1s)
		print(" ".join(rts))


def Pool_Comparison(results, sp=True):
	if sp: plt.subplot(rows,cols,get_gn())
	plt.title('ConvX Accuracy Against n Convolution & Max Pooling Layers')
	plt.ylabel('Accuracy')
	plt.xlabel('n Convolution & Max Pooling Layers')

	pools = [x for x in results if x["params"]["tid"] == 3 or (x["params"]["tid"] == 2 and x["params"]["id"] == "Conv2")]
	cl = [x["params"]["id"] for x in pools]
	accs = [x["test"]["accuracy"] for x in pools]
	f1s = [np.mean(x["test"]["f1_pc"]) for x in pools]
	rts = [ft(x["runtime"]) for x in pools]
	plt.bar(cl, accs)

	if sp:
		print("\nAccuracy of different pooling layers")
		print(cl)
		print(accs)
		print(f1s)
		print(" ".join(rts))


def Activation_Comparison_acc(results, sp=True):
	if sp: plt.subplot(rows,cols,get_gn())
	plt.title('Activation Function accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Activation Function')

	pools = [x for x in results if x["params"]["tid"] == 4 or x["params"]["id"] == "Conv2"]
	cl = [x["params"]["id"] for x in pools]
	accs = [round(x["test"]["accuracy"], 3) for x in pools]
	f1s = [round(np.mean(x["test"]["f1_pc"]), 3) for x in pools]
	rts = [ft(x["runtime"]) for x in pools]
	plt.bar(cl, accs)

	if sp:
		print("\nActivation function accuracy")
		print(cl)
		print(accs)
		print(f1s)
		print(" ".join(rts))


def Activation_Comparison_f1(results, sp=True):
	if sp: plt.subplot(rows,cols,get_gn())
	plt.title('Activation Function F1')
	plt.ylabel('F1')
	plt.xlabel('Activation Function')

	pools = [x for x in results if x["params"]["tid"] == 4 or x["params"]["id"] == "Conv2"]
	cl = [x["params"]["id"] for x in pools]
	f1s = [round(np.mean(x["test"]["f1_pc"]), 3) for x in pools]
	plt.bar(cl, f1s)



def channel_acc(results, sp=True):
	if sp: plt.subplot(rows,cols,get_gn())
	plt.title('Accuracy of different number of channels')
	plt.ylabel('Accuracy')
	plt.xlabel('Number of channels')

	res = [x for x in results if x["params"]["tid"] == 5 or (x["params"]["tid"] == 4 and x["params"]["model"] == "ReLU")]
	cl = [x["params"]["id"] for x in res]
	accs = [round(x["test"]["accuracy"], 3) for x in res]
	f1s = [round(np.mean(x["test"]["f1_pc"]), 3) for x in res]
	rts = [ft(x["runtime"]) for x in res]
	plt.bar(cl, accs)

	if sp:
		print("\nActivation function accuracy")
		print(cl)
		print(accs)
		print(f1s)
		print(" ".join(rts))


def epoch_acc(results, sp=True):
	if sp: plt.subplot(rows,cols,get_gn())
	plt.title('Accuracy and F1 Score of Model Against Number of Epochs')
	plt.ylabel('Accuracy')
	plt.xlabel('Number of Epochs')
	res = [x for x in results if x["params"]["tid"] == 6][-1]
	epochs = range(1, len(res["train"]["epochs"]) +1)
	accs = [x["accuracy"] for x in res["train"]["epochs"]]
	f1s = [x["f1"] for x in res["train"]["epochs"]]
	plot(epochs, accs, i=4, label="Accuracy")
	plot(epochs, f1s, i=5, label="F1 Score")
	plt.legend(loc="lower right")
	if sp:
		print("\nEpoch accuracy")
		print(epochs)
		print(accs)
		print(f1s)



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
	
	FCN_acc_v_hl(results, sp=False)
	plt.tight_layout()
	plt.savefig(p.format("FCN_acc_v_hl"))
	plt.clf()

	FCN_ttrain_v_hl(results, sp=False)
	plt.tight_layout()
	plt.savefig(p.format("FCN_ttrain_v_hl"))
	plt.clf()

	Conv_nc_v_acc(results, sp=False)
	plt.tight_layout()
	plt.savefig(p.format("Conv_nc_v_acc"))
	plt.clf()

	Activation_Comparison_acc(results, sp=False)
	plt.tight_layout()
	plt.savefig(p.format("Activation_Comparison_acc"))
	plt.clf()

	Activation_Comparison_f1(results, sp=False)
	plt.tight_layout()
	plt.savefig(p.format("Activation_Comparison_f1"))
	plt.clf()

	channel_acc(results, sp=False)
	plt.tight_layout()
	plt.savefig(p.format("channel_acc"))
	plt.clf()

	epoch_acc(results, sp=False)
	plt.tight_layout()
	plt.savefig(p.format("epoch_acc"))
	plt.clf()


	print("Images saved")


# If no filepath passed in use results.csv
if __name__ == "__main__":
	if len(sys.argv) > 1:
		main(str(sys.argv[1]))
	else:
		f_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv')
		main(f_path)
		