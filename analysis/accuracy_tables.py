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

# Parse arguments
opt = parser.parse_args()

import pickle as pkl

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
               "EEGChannelNet"]

f = open(opt.iv+"-full-accuracy.tex", "w")
f.write("\\begin{tabular}{@{}rrrrrrrr@{}}\n")
f.write("LSTM&$k$-NN&SVM&MLP&1D CNN&EEGNet&SyncNet&EEGChannelNet\\\\\n")
f.write("\\hline\n")
i = 0
for c in classifiers[0:len(classifiers)]:
    if c=="SVM":
        (accuracy, p_value) = pkl.load(
            open("run-100/"+c+"-100.pkl", "rb"))[
            (c,
             1,
             "imagenet40-1000",
             512,
             96,
             100)]
    else:
        (accuracy, p_value) = pkl.load(
            open("run-100/"+c+".pkl", "rb"))[
            (c,
             1,
             "imagenet40-1000",
             512,
             96)]
    if i!=0:
        f.write("&")
    f.write("%.1f"%(100*accuracy))
    f.write("\\%")
    f.write("\\sig" if p_value<opt.p_value else "\\notsig")
    i += 1
f.write("\\\\\n")
f.write("\\end{tabular}\n")
f.close()

classifiers = ["SVM", "CNN", "EEGNet"]

f = open(opt.iv+"-artifact-removal.tex", "w")
f.write("\\begin{tabular}{@{}rrr@{}}\n")
f.write("SVM&1D CNN&EEGNet\\\\\n")
f.write("\\hline\n")
i = 0
for c in classifiers[0:len(classifiers)]:
    try:
        if i!=0:
            f.write("&")
        (accuracy, p_value) = pkl.load(
            open("run-100/"+c+"-artifact-removal.pkl", "rb"))[
            (c,
             1,
             "imagenet40-1000",
             512,
             96)]
        f.write("%.1f"%(100*accuracy))
        f.write("\\%")
        f.write("\\sig" if p_value<opt.p_value else "\\notsig")
    except:
        pass
    i += 1
f.write("\\\\\n")
f.write("\\end{tabular}\n")
f.close()

classifiers = ["K-NN", "SVM", "MLP"]

f = open(opt.iv+"-pmtm.tex", "w")
f.write("\\begin{tabular}{@{}rrr@{}}\n")
f.write("$k$-NN&SVM&MLP\\\\\n")
f.write("\\hline\n")
i = 0
for c in classifiers[0:len(classifiers)]:
    try:
        if i!=0:
            f.write("&")
        (accuracy, p_value) = pkl.load(
            open("run-100/"+c+"-pmtm.pkl", "rb"))[
            (c,
             1,
             "imagenet40-1000",
             257,
             96)]
        f.write("%.1f"%(100*accuracy))
        f.write("\\%")
        f.write("\\sig" if p_value<opt.p_value else "\\notsig")
    except:
        pass
    i += 1
f.write("\\\\\n")
f.write("\\end{tabular}\n")
f.close()

classifiers = ["SVM", "CNN", "EEGNet"]

f = open(opt.iv+"-cross-subject.tex", "w")
f.write("\\begin{tabular}{@{}rrr@{}}\n")
f.write("SVM&1D CNN&EEGNet\\\\\n")
f.write("\\hline\n")
i = 0
for c in classifiers[0:len(classifiers)]:
    if i!=0:
        f.write("&")
    (accuracy, p_value) = pkl.load(
        open("run-100/"+c+"-cross-subject.pkl", "rb"))[
        (c,
         1,
         "cross-subject",
         512,
         96)]
    f.write("%.1f"%(100*accuracy))
    f.write("\\%")
    f.write("\\sig" if p_value<opt.p_value else "\\notsig")
    i += 1
f.write("\\\\\n")
f.write("\\end{tabular}\n")
f.close()
