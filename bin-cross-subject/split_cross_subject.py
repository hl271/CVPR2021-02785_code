import torch

eeg1 = torch.load("/fs/aux/qobi/eeg-datasets/spampinato/preprocessed/spampinato-1-image-rapid-event-no-bandpass.pth")
eeg2 = torch.load("/fs/aux/qobi/eeg-datasets/spampinato/preprocessed/spampinato-2-image-rapid-event-no-bandpass.pth")
eeg3 = torch.load("/fs/aux/qobi/eeg-datasets/spampinato/preprocessed/spampinato-3-image-rapid-event-no-bandpass.pth")
eeg4 = torch.load("/fs/aux/qobi/eeg-datasets/spampinato/preprocessed/spampinato-4-image-rapid-event-no-bandpass.pth")
eeg5 = torch.load("/fs/aux/qobi/eeg-datasets/spampinato/preprocessed/spampinato-5-image-rapid-event-no-bandpass.pth")
eeg6 = torch.load("/fs/aux/qobi/eeg-datasets/spampinato/preprocessed/spampinato-6-image-rapid-event-no-bandpass.pth")

torch.save({"images": (eeg1["images"]+
                       eeg2["images"]+
                       eeg3["images"]+
                       eeg4["images"]+
                       eeg5["images"]+
                       eeg6["images"]),
            "labels": (eeg1["labels"]+
                       eeg2["labels"]+
                       eeg3["labels"]+
                       eeg4["labels"]+
                       eeg5["labels"]+
                       eeg6["labels"]),
            "dataset": (eeg1["dataset"]+
                        eeg2["dataset"]+
                        eeg3["dataset"]+
                        eeg4["dataset"]+
                        eeg5["dataset"]+
                        eeg6["dataset"])},
           "/fs/aux/qobi/eeg-datasets/spampinato/preprocessed/spampinato-cross-subject-image-rapid-event-no-bandpass.pth")

torch.save({"splits": [
    {"train": [
        2000*j+k
        for j in range(6)
        if j!=i
        for k in range(2000)],
     "val": [
         2000*i+k
        for k in range(1000)],
     "test": [
         2000*i+k
        for k in range(1000, 2000)]}
    for i in range(6)]},
           "/fs/aux/qobi/eeg-datasets/spampinato/preprocessed/spampinato-cross-subject-image-rapid-event-no-bandpass_split.pth")
