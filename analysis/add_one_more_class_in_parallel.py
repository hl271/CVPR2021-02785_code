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
parser.add_argument("-gpus",
                    "--GPUindices",
                    default = 1,
                    type = int,
                    help = "gpu indices")
parser.add_argument("-k",
                    "--kind",
                    default = "no-model-file",
                    type = str,
                    help = "incremental/no-model-file")
parser.add_argument("-cmd",
                    "--cmd",
                    default = None,
                    type = str,
                    help = "run_all_two_class_jobs/run_init_add_one_more_class/run_all_add_one_more_class_jobs/run_after_add_one_more_class")
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
from Loader import EEGDataset, EEGDataset_window
import pickle as pkl
from analysis import *

def sort_results(path):
    folder = path
    pairs = []
    acc = []
    p_value = []
    files = os.listdir(folder)
    for f in files:
        class1 = f.split(".")[0].split("-")[-1]
        f = open(folder+"/"+f, "rb")
        pairs.append(class1)
        record = pkl.load(f)
        f.close()
        acc.append(record["acc"])
        p_value.append(record["p_value"])
    idx = sorted(range(len(acc)), key = lambda k: -acc[k])
    return pairs[idx[0]], acc[idx[0]], p_value[idx[0]]

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

if opt.iv=="image":
    lengths = [512]
    channels = [96]
    min_CNN = 200
    total = 40
elif opt.iv=="video":
    lengths = [4096]
    channels = [96]
    min_CNN = 256
    total = 12
full_length = lengths[0]
full_channel = channels[0]
length = lengths[0]
channel = channels[0]

def run_two_classes(gpu, folder, class1, class2):
    name = (folder+
            "/"+
            opt.classifier+
            "-class1-"+
            str(class1)+
            "-class2-"+
            str(class2)+
            ".pkl")
    try:
        f = open(name, "rb")
        record = pkl.load(f)
        f.close()
        accuracy = (record["acc"])
        p_value = (record["p_value"])
    except:
        torch.manual_seed(12)
        torch.cuda.manual_seed(12)
        np.random.seed(12)
        random.seed(12)
        accuracy, p_value = analysis(opt.iv,
                                     opt.offset,
                                     opt.fold,
                                     opt.eeg_dataset,
                                     opt.splits_path,
                                     total,
                                     [class1, class2],
                                     opt.classifier,
                                     opt.batch_size,
                                     gpu,
                                     length,
                                     channel,
                                     full_length,
                                     full_channel,
                                     min_CNN,
                                     opt,
                                     opt.kind)
        f = open(name, "wb")
        pkl.dump({"acc":accuracy, "p_value":p_value}, f)
        f.close()

def run_two_class_job(gpu, folder, job):
    class1 = job["class1"]
    class2 = job["class2"]
    run_two_classes(gpu, folder, class1, class2)

def all_two_class_jobs():
    jobs = []
    for class1 in [i for i in range(total)]:
        for class2 in range(class1+1, total):
            jobs.append({"class1": class1, "class2": class2})
    return jobs

def run_all_two_class_jobs():
    gpu = int(opt.GPUindex)
    gpus = int(opt.GPUindices)
    folder = opt.iv+"-"+opt.classifier+"-two-way"
    if gpu==0:
        if not os.path.exists(folder):
            os.makedirs(folder)
    jobs = all_two_class_jobs()
    for job in range(len(jobs)):
        if job%gpus==gpu:
	    run_two_class_job(gpu, folder, jobs[job])

def run_init_add_one_more_class():
    gpu = int(opt.GPUindex)
    if gpu==0:
        folder = opt.iv+"-"+opt.classifier+"-two-way"
        optimal_class, optimal_acc, optimal_p_value = sort_results(folder)
        base_classes = [int(optimal_class)]
        f = open("base-classes-"+opt.iv+"-"+opt.classifier+".pkl", "wb")
        pkl.dump(base_classes, f)
        f.close()
        stat = open("stat-"+opt.iv+"-"+opt.classifier+".txt", "w")
        stat.close()

def run_add_one_more_class(gpu, folder, base_classes, class1):
    name = (folder+
            "/"+
            opt.classifier+
            "-class1-"+
            str(class1)+
            ".pkl")
    try:
        f = open(name, "rb")
        record = pkl.load(f)
        f.close()
        accuracy = (record["acc"])
        p_value = (record["p_value"])
    except:
        torch.manual_seed(12)
        torch.cuda.manual_seed(12)
        np.random.seed(12)
        random.seed(12)
        accuracy, p_value = analysis(opt.iv,
                                     opt.offset,
                                     opt.fold,
                                     opt.eeg_dataset,
                                     opt.splits_path,
                                     total,
                                     base_classes+[class1],
                                     opt.classifier,
                                     opt.batch_size,
                                     gpu,
                                     length,
                                     channel,
                                     full_length,
                                     full_channel,
                                     min_CNN,
                                     opt,
                                     opt.kind)
        f = open(folder+
                 "/"+
                 opt.classifier+
                 "-class1-"+
                 str(class1)+
                 ".pkl",
                 "wb")
        pkl.dump({"acc":accuracy, "p_value":p_value}, f)
        f.close()

def run_add_one_more_class_job(gpu, folder, base_classes, job):
    class1 = job["class1"]
    run_add_one_more_class(gpu, folder, base_classes, class1)

def all_add_one_more_class_jobs(base_classes):
    jobs = []
    for class1 in [i for i in range(total) if i not in base_classes]:
        jobs.append({"class1": class1})
    return jobs

def run_all_add_one_more_class_jobs():
    f = open("base-classes-"+opt.iv+"-"+opt.classifier+".pkl", "rb")
    base_classes = pkl.load(f)
    f.close()
    folder = (opt.iv+
              "-"+
              opt.classifier+
              "-add/add-"+
              str(len(base_classes)+1))
    if not os.path.exists(folder):
        os.makedirs(folder)
    f = open("base-classes-"+opt.iv+"-"+opt.classifier+".pkl", "rb")
    base_classes = pkl.load(f)
    f.close()
    gpu = int(opt.GPUindex)
    gpus = int(opt.GPUindices)
    jobs = all_add_one_more_class_jobs(base_classes)
    for job in range(len(jobs)):
        if job%gpus==gpu:
	    run_add_one_more_class_job(gpu, folder, base_classes, jobs[job])

def run_after_add_one_more_class():
    gpu = int(opt.GPUindex)
    if gpu==0:
        f = open("base-classes-"+opt.iv+"-"+opt.classifier+".pkl", "rb")
        base_classes = pkl.load(f)
        f.close()
        folder = (opt.iv+
                  "-"+
                  opt.classifier+
                  "-add/add-"+
                  str(len(base_classes)+1))
        optimal_class, optimal_acc, optimal_p_value = sort_results(folder)
        base_classes.append(int(optimal_class))
        f = open("base-classes-"+opt.iv+"-"+opt.classifier+".pkl", "wb")
        pkl.dump(base_classes, f)
        f.close()
        stat = open("stat-"+opt.iv+"-"+opt.classifier+".txt", "a")
        line = "["
        k = 0
        for c in base_classes:
            line += str(c)
            if k!=len(base_classes)-1:
                line += ", "
            k += 1
        line += "] "+str(optimal_acc)+" "+str(optimal_p_value)+"\n"
        stat.write(line)
        stat.close()

if opt.cmd=="run_all_two_class_jobs":
    run_all_two_class_jobs()
elif opt.cmd=="run_init_add_one_more_class":
    run_init_add_one_more_class()
elif opt.cmd=="run_all_add_one_more_class_jobs":
    run_all_add_one_more_class_jobs()
elif opt.cmd=="run_after_add_one_more_class":
    run_after_add_one_more_class()
else:
    raise RuntimeError("invalid cmd")
