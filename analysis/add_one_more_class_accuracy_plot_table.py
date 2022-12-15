import argparse
parser = argparse.ArgumentParser(description = "Template")
# Dataset options
parser.add_argument("-iv",
                    "--iv",
                    help = "image/video",
                    type = str,
                    required = True)
parser.add_argument("-p",
                    "--p_value",
                    help = "p value for significance",
                    type = float,
                    required = True)
parser.add_argument("-s",
                    "--samples",
                    help = "number of samples",
                    type = int,
                    required = True)

# Parse arguments
opt = parser.parse_args()

import matplotlib.pyplot as plt
import numpy as np
from p_values import *

if opt.iv=="image":
    num_classes = 40
elif opt.iv=="video":
    num_classes = 12

classifiers = ["LSTM",
               "K-NN",
               "SVM",
               "MLP",
               "CNN",
               "EEGNet",
               "SyncNet",
               "EEGChannelNet",
               "significance",
               "chance"]

acc = np.zeros((len(classifiers), num_classes-1))
acc[:, :] = None

i = 0
for c in classifiers:
    if c=="significance":
        for k in range(num_classes-1):
            acc[i, k] = significance(
                opt.p_value, ((k+2)*opt.samples)/num_classes, k+2)
    elif c=="chance":
        for k in range(num_classes-1):
            acc[i, k] = 1.0/(k+2)
    else:
        try:
            stat = open("run-100/stat-"+opt.iv+"-"+c+".txt", "r")
            lines = stat.readlines()
            stat.close()
            k = 0
            for v in lines:
                acc[i, k] = float(v.split("]")[1].split()[0])
                k += 1
        except:
            pass
    i += 1

# https://matplotlib.org/3.1.0/gallery/color/named_colors.html
colors = ["b", "g", "r", "c", "m", "y", "salmon", "lime", "grey", "k"]
# "teal"
# https://matplotlib.org/api/markers_api.html#module-matplotlib.markers
# https://matplotlib.org/gallery/lines_bars_and_markers/linestyles.html
for i in range(len(classifiers)):
    plt.plot(range(2, num_classes+1),
             acc[i, :]*100,
             color = colors[i],
             marker = "o",
             linestyle = "solid",
             label = classifiers[i],
             markersize = 3)
plt.ylim(0, 100)
plt.legend(loc = "upper right")
plt.xlabel("Number of classes")
plt.ylabel("Accuracy (%)")
plt.savefig(opt.iv+"-accuracy.png")

f = open(opt.iv+"-accuracy.tex", "w")
f.write("\\begin{tabular}{@{}r|rrrrrrrr@{}}\n")
f.write("&\\multicolumn{8}{c}{accuracy}\\\\\n")
f.write("number of classes&LSTM&$k$-NN&SVM&MLP&1D CNN&EEGNet&SyncNet&EEGChannelNet\\\\\n")
f.write("\\hline\n")
for k in range(num_classes-1):
    f.write(str(k+2))
    i = 0
    for c in classifiers[0:len(classifiers)-2]:
        f.write("&")
        f.write("%.1f"%(100*acc[i, k]))
        f.write("\\%")
        f.write("\\sig" if acc[i, k]>acc[8, k] else "\\notsig")
        i += 1
    f.write("\\\\\n")
f.write("\\end{tabular}\n")
f.close()
