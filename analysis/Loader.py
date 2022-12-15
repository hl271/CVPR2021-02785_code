import torch
import random
import copy

class EEGDataset:

    def __init__(self, iv, eeg_signals_path, classifier, map_idx = None):
        self.iv = iv
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        self.data = loaded["dataset"]
        try:
            self.means = loaded["means"]
            self.stddevs = loaded["stddevs"]
        except:
            pass
        self.classifier = classifier
        self.size = len(self.data) # 40000
        if classifier == "SCNN":
            for i in range(len(self.data)):
                C, L = self.data[i]["eeg"].shape # (channels, samples)
                data_mapped = torch.zeros(12, 15, L)
                for c in range(C):
                    data_mapped[map_idx[c][0], map_idx[c][1], :] = (
                        self.data[i]["eeg"][c, :])
            self.data[i]["eeg"] = data_mapped

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        # Process EEG
        try:
            eeg = ((self.data[i]["eeg"].float()-self.means)/self.stddevs).t()
        except:
            if self.classifier=="SCNN":
                eeg = self.data[i]["eeg"].float().permute(2,0,1)
            else:
                eeg = self.data[i]["eeg"].float().t()
        if self.iv=="spampinato" or self.iv=="image":
            if self.classifier=="SVM":
                if eeg.size(0)==257:
                    eeg = eeg[:,:]
                else:
                    eeg = eeg[::2,:]
            else:
                eeg = eeg[:,:]
        elif self.iv=="video":
            if self.classifier=="SVM":
                eeg = eeg[::2,:]
            else:
                eeg = eeg[:,:]
        # Get label
        label = self.data[i]["label"]
        return eeg, label

def randomOffset(iv, offset, dataset, length):
    totalNum = len(dataset)
    for i in range(totalNum):
        if iv=="spampinato" or iv=="image":
            if offset is None:
                start = random.randint(0, 512-length)
            else:
                start = offset
        elif iv=="video":
            start = random.randint(0, 1024-length)
        try:
            dataset[i]["eeg"] = dataset[i]["eeg"][:, start:start+length]
        except:
            pass
    return dataset

class EEGDataset_window:

    def __init__(self,
                 iv,
                 offset,
                 eeg_signals_path,
                 classifier,
                 length,
                 map_idx = None):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        self.data = loaded["dataset"]
        self.fakedata = copy.deepcopy(self.data)
        self.fakedata = randomOffset(iv, offset, self.fakedata, length)
        try:
                self.means = loaded["means"]
                self.stddevs = loaded["stddevs"]
        except:
            pass
            self.classifier = classifier
            self.size = len(self.data)
        if classifier == "SCNN":
                for i in range(len(self.data)):
                    C, L = self.data[i]["eeg"].shape
                    data_mapped = torch.zeros(12, 15, L)
                    for c in range(C):
                        data_mapped[map_idx[c][0], map_idx[c][1], :] = (
                            self.data[i]["eeg"][c, :])
        self.data[i]["eeg"] = data_mapped

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        # Process EEG
        try:
            eeg = ((self.fakedata[i]["eeg"].float()-
                    self.means)/self.stddevs).t()
        except:
            if self.classifier=="SCNN":
                eeg = self.fakedata[i]["eeg"].float().permute(2,0,1)
            else:
                eeg = self.fakedata[i]["eeg"].float().t()
        # Get label
        label = self.fakedata[i]["label"]
        return eeg, label

class Splitter:

    def __init__(
            self, iv, dataset, splits_path, classes, split_num, split_name):
        # Load split
        loaded = torch.load(splits_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        if iv=="spampinato":
            self.split_idx = [i for i in self.split_idx
                              if dataset.data[i]["label"] in classes
			      and 480<=dataset.data[i]["eeg"].size(1)]
        else:
            self.split_idx = [i for i in self.split_idx
                              if dataset.data[i]["label"] in classes]
        # Compute size
        self.size = len(self.split_idx)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        # Get sample from dataset
        return self.split_idx[i]

class Splitter_nn:

    def __init__(self,
                 iv,
                 dataset,
                 splits_path,
                 classes,
                 split_num,
                 split_name,
                 relabel):
        # Set EEG dataset
        self.dataset = dataset
        self.classes = classes
        self.relabel = relabel
        # Load split
        loaded = torch.load(splits_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        if iv=="spampinato":
            self.split_idx = [i for i in self.split_idx
                              if dataset.data[i]["label"] in classes
			      and 480<=dataset.data[i]["eeg"].size(1)]
        else:
            self.split_idx = [i for i in self.split_idx
                              if dataset.data[i]["label"] in classes]
        # Compute size
        self.size = len(self.split_idx)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]
        if self.relabel:
            label = self.classes.index(label)
        return eeg, label
