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
	# plt.suptitle("Runtime: {:.3f}, Train Set: {}, Test Set: {}".format(
											# sum([x["run"]["runtime"] for x in results]),
											# results[0]["param"]["n_train"],
											# results[0]["param"]["n_test"]))
	plt.subplots_adjust(bottom=0.08, top=0.9, left=0.08, right=0.95, wspace=0.2, hspace=0.4)
	rows = 2
	cols = 2


	FCN_acc_v_hl(results)
	FCN_acc_pc_v_hl(results)
	FCN_ttrain_v_hl(results)


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
	plot(hl, accs, 0)



def FCN_acc_pc_v_hl(results, sp=True):
	if sp: plt.subplot(rows,cols,get_gn())
	plt.title('FCN Accuracy Per Class Against n Hidden Layers')
	plt.ylabel('Accuracy')
	plt.xlabel('n')

	fcns = [x for x in results if x["params"]["model"] == "FCN"]
	hl = [x["params"]["hidden_layers"] for x in fcns]
	for ci in range(10):
		accs = [x["test"]["accuracy_pc"][ci] for x in fcns]
		plot(hl, accs, ci, label=get_label(ci))

	plt.legend(loc="lower right")


def FCN_ttrain_v_hl(results, sp=True):
	if sp: plt.subplot(rows,cols,get_gn())
	plt.title('FCN Train time Against n Hidden Layers')
	plt.ylabel('Train time / seconds')
	plt.xlabel('n')

	fcns = [x for x in results if x["params"]["model"] == "FCN"]
	hl = [x["params"]["hidden_layers"] for x in fcns]
	times = [x["train"]["runtime"] for x in fcns]
	plot(hl, times, 0)



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

	FCN_acc_pc_v_hl(results, sp=False)
	plt.tight_layout()
	plt.savefig(p.format("FCN_acc_pc_v_hl"))
	plt.clf()

	FCN_ttrain_v_hl(results, sp=False)
	plt.tight_layout()
	plt.savefig(p.format("FCN_ttrain_v_hl"))
	plt.clf()

	print("Images saved")


# If no filepath passed in use results.csv
if __name__ == "__main__":
	if len(sys.argv) > 1:
		main(str(sys.argv[1]))
	else:
		f_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv')
		main(f_path)
		