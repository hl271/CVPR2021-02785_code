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
import pickle as pkl

if opt.iv=="image":
    num_classes = 40
elif opt.iv=="video":
    num_classes = 12

classifiers = ["SVM",
               "CNN",
               "EEGNet",
               "significance",
               "chance"]

acc = np.zeros((len(classifiers), 10))
acc[:, :] = None

i = 0
for c in classifiers:
    if c=="significance":
        for k in range(10, 110, 10):
            acc[i, k/10-1] = significance(
                opt.p_value, (k*opt.samples)/100, num_classes)
    elif c=="chance":
        for k in range(10, 110, 10):
            acc[i, k/10-1] = 1.0/num_classes
    else:
        try:
            for k in range(10, 110, 10):
                (accuracy, p_value) = pkl.load(
                    open("run-100/"+c+"-"+str(k)+".pkl", "rb"))[
                        (c,
                         1,
                         "imagenet40-1000",
                         512,
                         96,
                         k)]
                acc[i, k/10-1] = accuracy
        except:
            pass
    i += 1

# https://matplotlib.org/3.1.0/gallery/color/named_colors.html
colors = ["r", "m", "y", "grey", "k"]
# https://matplotlib.org/api/markers_api.html#module-matplotlib.markers
# https://matplotlib.org/gallery/lines_bars_and_markers/linestyles.html
for i in range(len(classifiers)):
    plt.plot(range(10, 110, 10),
             acc[i, :]*100,
             color = colors[i],
             marker = "o",
             linestyle = "solid",
             label = classifiers[i],
             markersize = 3)
plt.xlim(10, 100)
plt.ylim(0, 11)
plt.legend(loc = "upper right")
plt.xlabel("Fraction of dataset (%)")
plt.ylabel("Accuracy (%)")
plt.savefig(opt.iv+"-partial-accuracy.png")

f = open(opt.iv+"-partial-accuracy.tex", "w")
f.write("\\begin{tabular}{@{}r|rrr@{}}\n")
f.write("&\\multicolumn{3}{c}{accuracy}\\\\\n")
f.write("fraction of dataset&SVM&1D CNN&EEGNet\\\\\n")
f.write("\\hline\n")
i = 0
for k in range(10, 110, 10):
    f.write(str(k))
    f.write("\\%")
    j = 0
    for c in classifiers[0:len(classifiers)-2]:
        f.write("&")
        f.write("%.1f"%(100*acc[j, i]))
        f.write("\\%")
        f.write("\\sig" if acc[j, i]>acc[3, i] else "\\notsig")
        j += 1
    f.write("\\\\\n")
    i += 1
f.write("\\end{tabular}\n")
f.close()
