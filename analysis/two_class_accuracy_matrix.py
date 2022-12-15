import argparse
parser = argparse.ArgumentParser(description = "Template")
parser.add_argument("-iv",
                    "--iv",
                    help = "image/video",
                    type = str,
                    required = True)
parser.add_argument("-c",
                    "--classifier",
                    required = True,
                    help = "K-NN/SVM/LSTM/LSTM1/LSTM2/LSTM3/LSTM4/MLP/CNN/SCNN/EEGNet/SyncNet/EEGChannelNet")
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
opt = parser.parse_args()

import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
from p_values import *
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

if opt.iv=="image":
    class_name = ["German_shepherd",
                  "Egyptian_cat",
                  "lycaenid",
                  "sorrel",
                  "capuchin",
                  "African_elephant",
                  "giant_panda",
                  "anemone_fish",
                  "airliner",
                  "broom",
                  "canoe",
                  "cellular_telephone",
                  "coffee_mug",
                  "convertible",
                  "desktop_computer",
                  "digital_watch",
                  "electric_guitar",
                  "electric_locomotive",
                  "espresso_maker",
                  "folding_chair",
                  "golf_ball",
                  "grand_piano",
                  "iron",
                  "jack-o-lantern",
                  "mailbag",
                  "missile",
                  "mitten",
                  "mountain_bike",
                  "mountain_tent",
                  "pajama",
                  "parachute",
                  "pool_table",
                  "radio_telescope",
                  "reflex_camera",
                  "revolver",
                  "running_shoe",
                  "banana",
                  "pizza",
                  "daisy",
                  "bolete"]

else:
    class_name = ["AnswerPhone",
                  "DriveCar",
                  "Eat",
                  "FightPerson",
                  "GetOutCar",
                  "HandShake",
                  "HugPerson",
                  "Kiss",
                  "Run",
                  "SitDown",
                  "SitUp",
                  "StandUp"]

n_classes = len(class_name)
accuracy_matrix = np.zeros((n_classes, n_classes))
folder = "run-100/"+opt.iv+"-"+opt.classifier+"-two-way"

for i in range(n_classes):
    for j in range(i, n_classes):
        if i==j:
            accuracy_matrix[i, j] = 1.0
        else:
            name = (folder+
                "/"+
                opt.classifier+
                "-class1-"+
                str(i)+
                "-class2-"+
                str(j)+
                ".pkl")
            acc = pkl.load(open(name, "rb"))["acc"]
            accuracy_matrix[i, j] = acc
            accuracy_matrix[j, i] = acc

s = significance(opt.p_value, (2*opt.samples)/n_classes, 2)
significant_cmap = cm.get_cmap("Greens", 65536)(np.linspace(0, 1, 65536))
insignificant_cmap = cm.get_cmap("Reds", 65536)(np.linspace(0, 1, 65536))
thresh = int((2.0*s-1.0)*65536)
colors = np.concatenate((insignificant_cmap[:thresh],
                         significant_cmap[thresh:]))
colors[0, :] = 1.0
cmap = ListedColormap(colors)

accuracy_matrix_plot = copy.deepcopy(accuracy_matrix)
accuracy_matrix_plot[accuracy_matrix_plot<0.5] = 0.5
fig, ax = plt.subplots(figsize = (n_classes/2, n_classes/2))
ax.xaxis.tick_top()
im = ax.imshow(accuracy_matrix_plot, vmin = 0.5, vmax = 1.0, cmap = cmap)
ax.figure.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04)
ax.set(xticks = np.arange(accuracy_matrix.shape[1]),
       yticks = np.arange(accuracy_matrix.shape[0]),
       xticklabels = class_name,
       yticklabels = class_name)

plt.setp(ax.get_xticklabels(),
         rotation = 45,
         ha = "left",
         va = "bottom",
         rotation_mode = "anchor")
fmt = ".2f"
thresh = 0.75
for i in range(accuracy_matrix.shape[0]):
    for j in range(accuracy_matrix.shape[1]):
        if accuracy_matrix_plot[i,j]==0.5:
            continue
        ax.text(j, i, format(accuracy_matrix[i, j], fmt),
                ha = "center", va = "center",
                color = ("white" if accuracy_matrix_plot[i, j]>thresh
                         else "black"))
fig.tight_layout()
save_root = "two-class-accuracy-matrix"
if not os.path.exists(save_root):
    os.makedirs(save_root)
plt.savefig(save_root+"/"+opt.classifier+"-"+opt.iv+".png")
