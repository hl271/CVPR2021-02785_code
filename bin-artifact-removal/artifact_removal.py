import torch
eeg = torch.load("/fs/aux/qobi/eeg-datasets/imagenet40-1000/preprocessed-no-z-score/imagenet40-1000-1.pth")["dataset"]

swings = [
    (eeg[t]["eeg"].max(1)[0]-eeg[t]["eeg"].min(1)[0]).max(0)[0].item()
    for t in range(len(eeg))]

threshold = sorted(swings)[int(0.98*len(swings))]

print "candidate threshold: %f"%threshold

threshold = 600.0

print "threshold: %f"%threshold

trials_to_discard = [t for t in range(len(swings)) if (swings[t]>=threshold)]

print "%d trials discarded"%len(trials_to_discard)

split = torch.load("/fs/aux/qobi/eeg-datasets/imagenet40-1000/preprocessed/imagenet40-1000-1_split.pth")

for s in range(len(split["splits"])):
    for kind in split["splits"][s]:
        split["splits"][s][kind] = [
            t
            for t in split["splits"][s][kind]
            if t not in trials_to_discard]

torch.save(split, "/fs/aux/qobi/eeg-datasets/imagenet40-1000/preprocessed-no-z-score/imagenet40-1000-1_split_artifact_removal.pth")
