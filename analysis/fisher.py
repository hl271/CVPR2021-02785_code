import numpy as np
import torch

def fisher(dataset, loader, total, threshold):
    eeg = []
    eeg_label = []
    for idx in loader:
        input, target = dataset[idx]
        eeg.append(input)
        eeg_label.append(target)
    totalNum = len(eeg)
    time, channel = eeg[0].shape
    data = np.zeros((totalNum, time, channel))
    label = np.zeros((totalNum, time, 1))
    for i in range(totalNum):
        data[i, :, :] = eeg[i].float().numpy()
	for j in range(time):
            label[i, j, 0] = eeg_label[i]
    del eeg, eeg_label
    data = np.reshape(data, (-1, channel))
    label = np.reshape(label, (-1, 1))
    mean = np.mean(data, axis = 0)
    for i in range(data.shape[0]):
        data[i, :] = data[i, :]-mean
    per_class_mean = np.zeros((total, channel))
    per_class_std = np.zeros((total, channel))
    per_class_samples = np.zeros((total, 1))
    for i in range(data.shape[0]):
        classi = int(label[i, 0])
        per_class_mean[classi, :] = per_class_mean[classi, :]+data[i, :]
	per_class_samples[classi, 0] += 1
    for i in range(total):
        per_class_mean[i, :] = per_class_mean[i, :]/per_class_samples[i, 0]
    for i in range(data.shape[0]):
        classi = int(label[i, 0])
        per_class_std[classi, :] = (per_class_std[classi, :]+
                                    np.square((data[i, :]-
                                               per_class_mean[classi, :])))
    for i in range(total):
        per_class_std[i, :] = (np.sqrt(per_class_std[i, :]/
                                       per_class_samples[i, 0]))
    score = np.zeros(channel)
    for i in range(channel):
        v = 0.0
        n = 0.0
        for j in range(total):
            v += per_class_samples[j, 0]*per_class_std[j, i]*per_class_std[j, i]
            n += (per_class_samples[j, 0]*
                  per_class_mean[j, i]*
                  per_class_mean[j, i])
        if v==0:
            score[i] = 0
        else:
            score[i] = n/v
    idx_sort = np.argsort(score)
    idx_sort = idx_sort[::-1]
    idx_sort = idx_sort[0:threshold]
    idx_sort = torch.LongTensor(idx_sort.tolist())
    return idx_sort
