from torch.utils.data import DataLoader
from Loader import EEGDataset, EEGDataset_window, Splitter, Splitter_nn
from classifiers import *
from fisher import fisher
from p_values import *

# Map is coordinate matrix of channels
Map = [[ 0,  0,  0,  0,  0,  0,  1,  2, 33,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  4,  3, 34, 35, 36,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  5,  6,  7,  8, 39, 38, 37,  0,  0,  0,  0],
       [ 0,  0,  0, 12, 11, 10,  9, 40, 41, 42, 43, 44,  0,  0,  0],
       [ 0, 13, 14, 15, 16, 17, 18, 19, 50, 49, 48, 47, 46, 45,  0],
       [26, 25, 24, 23, 22, 21, 20, 51, 52, 53, 54, 55, 56, 57, 58],
       [ 0, 27, 28, 29, 30, 31, 32, 71, 64, 63, 62, 61, 60, 59,  0],
       [ 0, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78,  0],
       [ 0,  0,  0,  0,  0, 83, 82, 81, 80, 79,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0, 85, 86, 87, 88, 89,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0, 84, 94, 93, 92, 90,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0, 95, 96, 91,  0,  0,  0,  0,  0,  0]]

#idx is dictionary storing coordinates of each channel
idx = {}

for i in range(96):
        for m in range(12):
                for n in range(15):
                        if Map[m][n]==i+1:
                                idx[i] = (m,n)

def analysis(iv,
             offset,
             fold,
             eeg_dataset,
             splits_path,
             total,
             classes,
             classifier,
             batch_size,
             GPUindex,
             length, # 512
             channel, # 96
             full_length,
             full_channel,
             min_CNN,
             opt,
             kind):
    val, test, samples = 0.0, 0.0, 0
    for split_num in range(fold):
        model_path = (iv+
                      "-"+
                      classifier+
                      "-"+
                      str(length)+
                      "-"+
                      str(channel)+
                      "-"+
                      str(split_num))
        if channel==full_channel:
            do_fisher = False
        else:
            do_fisher = True
        # Load dataset
        if length!=full_length:
            dataset = EEGDataset_window(iv,
                                        offset,
                                        eeg_dataset,
                                        classifier,
                                        length,
                                        map_idx = idx)
        else:
            dataset = EEGDataset(
                iv, eeg_dataset, classifier, map_idx = idx)
        if classifier=="K-NN" or classifier=="SVM":
            # Create loaders for KNN/SVM
            loaders = {split: Splitter(
                    iv, dataset, splits_path, classes,  split_num, split)
                       for split in ["train", "val", "test"]}
            if do_fisher:
                channel_idx = fisher(dataset, loaders["train"], total, channel)
            else:
                channel_idx = None
        else:
            # Create loaders for LSTM/MLP/CNN/SCNN/EEGNet/SyncNet/EEGChannelNet
            if kind=="from-scratch":
                relabel = True
            if kind=="incremental":
                relabel = False
            if kind=="no-model-file":
                relabel = True
            loaders = {split: DataLoader(
                Splitter_nn(iv,
                            dataset,
                            splits_path,
                            classes,
                            split_num,
                            split,
                            relabel),
                batch_size = batch_size,
                drop_last = False,
                shuffle = True)
                   for split in ["train", "val", "test"]}
            if do_fisher:
                loader_fisher = Splitter(
                    iv, dataset, splits_path, classes, split_num, "train")
                channel_idx = fisher(dataset, loader_fisher, total, channel)
            else:
                channel_idx = None
        if classifier=="K-NN":
            k = 7
            accuracy_val, counts_val = classifier_KNN(
                dataset,
                loaders["train"],
                loaders["val"],
                k,
                channel_idx)
            accuracy_test, counts_test = classifier_KNN(
                dataset,
                loaders["train"],
                loaders["test"],
                k,
                channel_idx)
        elif classifier=="SVM":
            nonclasses = [i for i in range(total) if i not in classes]
            if kind=="from-scratch":
                accuracy_val, accuracy_test, counts_val, counts_test = classifier_SVM(
                    dataset,
                    loaders["train"],
                    loaders["val"],
                    loaders["test"],
                    channel_idx,
                    nonclasses,
                    None,
                    model_path)
            if kind=="incremental":
                accuracy_val, accuracy_test, counts_val, counts_test = classifier_SVM(
                    dataset,
                    loaders["train"],
                    loaders["val"],
                    loaders["test"],
                    channel_idx,
                    nonclasses,
                    model_path,
                    None)
            if kind=="no-model-file":
                accuracy_val, accuracy_test, counts_val, counts_test = classifier_SVM(
                    dataset,
                    loaders["train"],
                    loaders["val"],
                    loaders["test"],
                    channel_idx,
                    nonclasses,
                    None,
                    None)
        else:
            if classifier=="LSTM":
                if kind=="from-scratch":
                    output_size = 128
                if kind=="incremental":
                    output_size = 128
                if kind=="no-model-file":
                    output_size = 128
                net = classifier_LSTM(
                    True,
                    input_size = channel,
                    lstm_layers = 1,
                    lstm_size = 128,
                    output1_size = 128,
                    output2_size = None,
                    GPUindex = GPUindex)
            elif classifier=="LSTM1":
                if kind=="from-scratch":
                    output_size = 128
                if kind=="incremental":
                    output_size = 128
                if kind=="no-model-file":
                    output_size = 128
                net = classifier_LSTM(
                    False,
                    input_size = channel,
                    lstm_layers = 1,
                    lstm_size = 128,
                    output1_size = 128,
                    output2_size = None,
                    GPUindex = GPUindex)
            elif classifier=="LSTM2":
                if kind=="from-scratch":
                    output_size = len(classes)
                if kind=="incremental":
                    output_size = total
                if kind=="no-model-file":
                    output_size = len(classes)
                net = classifier_LSTM(
                    False,
                    input_size = channel,
                    lstm_layers = 1,
                    lstm_size = 128,
                    output1_size = output_size,
                    output2_size = None,
                    GPUindex = GPUindex)
            elif classifier=="LSTM3":
                if kind=="from-scratch":
                    output_size = len(classes)
                if kind=="incremental":
                    output_size = total
                if kind=="no-model-file":
                    output_size = len(classes)
                net = classifier_LSTM(
                    True,
                    input_size = channel,
                    lstm_layers = 1,
                    lstm_size = 128,
                    output1_size = output_size,
                    output2_size = None,
                    GPUindex = GPUindex)
            elif classifier=="LSTM4":
                if kind=="from-scratch":
                    output_size = len(classes)
                if kind=="incremental":
                    output_size = total
                if kind=="no-model-file":
                    output_size = len(classes)
                net = classifier_LSTM(
                    True,
                    input_size = channel,
                    lstm_layers = 1,
                    lstm_size = 128,
                    output1_size = 128,
                    output2_size = output_size,
                    GPUindex = GPUindex)
            elif classifier=="MLP":
                if kind=="from-scratch":
                    output_size = len(classes)
                if kind=="incremental":
                    output_size = total
                if kind=="no-model-file":
                    output_size = len(classes)
                net = classifier_MLP(
                        input_size = channel*length, output_size = output_size)
            elif classifier=="CNN":
                if length<min_CNN:
                    break
                if kind=="from-scratch":
                    output_size = len(classes)
                if kind=="incremental":
                    output_size = total
                if kind=="no-model-file":
                    output_size = len(classes)
                net = classifier_CNN(
                    in_channel = channel,
                    num_points = length,
                        output_size = output_size)
            elif classifier=="SCNN":
                if length<min_CNN:
                    break
                if kind=="from-scratch":
                    output_size = len(classes)
                if kind=="incremental":
                    output_size = total
                if kind=="no-model-file":
                    output_size = len(classes)
                net = classifier_SCNN(
                    in_channel = channel,
                    num_points = length,
                    output_size = output_size)
            elif classifier=="EEGNet":
                if length<min_CNN:
                    break
                if kind=="from-scratch":
                    output_size = len(classes)
                if kind=="incremental":
                    output_size = total
                if kind=="no-model-file":
                    output_size = len(classes)
                net = classifier_EEGNet(channel, length)
            elif classifier=="SyncNet":
                if length<min_CNN:
                    break
                if kind=="from-scratch":
                    output_size = len(classes)
                if kind=="incremental":
                    output_size = total
                if kind=="no-model-file":
                    output_size = len(classes)
                net = classifier_SyncNet(channel, length)
            elif classifier=="EEGChannelNet":
                if length<min_CNN:
                    break
                if kind=="from-scratch":
                    output_size = len(classes)
                if kind=="incremental":
                    output_size = total
                if kind=="no-model-file":
                    output_size = len(classes)
                net = classifier_EEGChannelNet(channel, length)
            nonclasses = [i for i in range(output_size) if i not in classes]
            if kind=="from-scratch":
                accuracy_val, accuracy_test, counts_val, counts_test = net_trainer(
                        net,
                        loaders,
                        opt,
                        channel_idx,
                        nonclasses,
                        None,
                        True,
                        model_path)
            if kind=="incremental":
                accuracy_val, accuracy_test, counts_val, counts_test = net_trainer(
                        net,
                        loaders,
                        opt,
                        channel_idx,
                        nonclasses,
                        model_path,
                        True,
                        None)
            if kind=="no-model-file":
                accuracy_val, accuracy_test, counts_val, counts_test = net_trainer(
                        net,
                        loaders,
                        opt,
                        channel_idx,
                        nonclasses,
                        None,
                        True,
                        None)
        val += accuracy_val
        test += accuracy_test
        samples += counts_val+counts_test
    accuracy = (val+test)/(fold*2)
    return accuracy, p_value(accuracy, samples, len(classes))
