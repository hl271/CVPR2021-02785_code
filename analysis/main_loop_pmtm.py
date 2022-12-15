from __future__ import division
import argparse
parser = argparse.ArgumentParser(description = "Template")
# Dataset options
parser.add_argument("-iv",
                    "--iv",
                    help = "image/video",
                    type = str,
                    required = True)
parser.add_argument("-off",
                    "--offset",
                    default = None,
                    type = int,
                    help = "offset")
parser.add_argument("-rf",
                    "--results-file",
                    default = "results.plk",
                    help = "results file",
                    type = str,
                    required = False)
parser.add_argument("-s",
                    "--subject",
                    default = 0,
                    type = int,
                    help = "subject")
parser.add_argument("-r",
                    "--run",
                    default = "none",
                    type = str,
                    help = "run")
parser.add_argument("-ed",
                    "--eeg-dataset",
                    help = "EEG dataset path")
parser.add_argument("-sp",
                    "--splits-path",
                    help = "splits path")
parser.add_argument("-f",
                    "--fold",
                    default = 5,
                    help = "number of folds",
                    type = int,
                    required = False)
# Training options
parser.add_argument("-b",
                    "--batch_size",
                    default = 16,
                    type = int,
                    help = "batch size")
parser.add_argument("-o",
                    "--optim",
                    default = "Adam",
                    help = "optimizer")
parser.add_argument("-lr",
                    "--learning-rate",
                    default = 0.001,
                    type = float,
                    help = "learning rate")
parser.add_argument("-lrdb",
                    "--learning-rate-decay-by",
                    default = 0.5,
                    type = float,
                    help = "learning rate decay factor")
parser.add_argument("-lrde",
                    "--learning-rate-decay-every",
                    default = 10,
                    type = int,
                    help = "learning rate decay period")
parser.add_argument("-e",
                    "--epochs",
                    default = 100,
                    type = int,
                    help = "training epochs")
parser.add_argument("-gpu",
                    "--GPUindex",
                    default = 0,
                    type = int,
                    help = "gpu index")
parser.add_argument("-k",
                    "--kind",
                    default = "incremental",
                    type = str,
                    help = "from-scratch/incremental/no-model-file")
# Backend options
parser.add_argument("--no-cuda",
                    default = False,
                    help = "disable CUDA",
                    action = "store_true")
parser.add_argument("-c",
                    "--classifier",
                    required = True,
                    help = "K-NN/SVM/LSTM/LSTM1/LSTM2/LSTM3/LSTM4/MLP/CNN/SCNN/EEGNet/SyncNet/EEGChannelNet")

# Parse arguments
opt = parser.parse_args()

# Imports
import os
import random
import torch
import numpy as np
import pickle as pkl
from analysis import *

torch.manual_seed(12)
torch.cuda.manual_seed(12)
np.random.seed(12)
random.seed(12)
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

if opt.iv=="image":
    lengths = [257]
    channels = [96]
    min_CNN = 200
    n_classes = 40
elif opt.iv=="video":
    lengths = [4096]
    channels = [96]
    min_CNN = 256
    n_classes = 12
full_length = lengths[0]
full_channel = channels[0]
classes = range(n_classes)

for length in lengths:
    for channel in channels:
        accuracy, p_value = analysis(opt.iv,
                                     opt.offset,
                                     opt.fold,
                                     opt.eeg_dataset,
                                     opt.splits_path,
                                     n_classes,
                                     classes,
                                     opt.classifier,
                                     opt.batch_size,
                                     opt.GPUindex,
                                     length,
                                     channel,
                                     full_length,
                                     full_channel,
                                     min_CNN,
                                     opt,
                                     opt.kind)
        print((opt.classifier,
               opt.subject,
               opt.run,
               length,
               channel,
               accuracy,
               p_value))
        exists = os.path.isfile(opt.results_file)
        if exists:
            f = open(opt.results_file, "rb")
            results = pkl.load(f)
            f.close()
        else:
            results = {}
        results[(opt.classifier,
                 opt.subject,
                 opt.run,
                 length,
                 channel)] = (accuracy, p_value)
        f = open(opt.results_file, "wb")
        pkl.dump(results, f)
        f.close()
