import mne
import os
import tempfile

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
data_path = os.path.join(parent_dir, 'data', 'imagenet40-1000-')
design_path = os.path.join(parent_dir, 'design', 'run')
tmp_path = tempfile.gettempdir()
tmp_path = os.path.join(tmp_path, 'imagenet40-1000', 'imagenet40-1000-')
band = 0
notch = 0

for subject in range(1, 2):
    out = r'{0}{1}\''.format(tmp_path, subject)
    print(out)
    for run in range(100):
        bdf = '{0}{1}-{2:2d}.bdf'.format(data_path, subject, run)
        print(bdf) if run ==15 else None
        stim = '{0}-{1:2d}.txt'.format(design_path, run)
        print(stim) if run ==15 else None
        trim = run==14
        # read_EEG(bdf, band, notch, 400, trim)
        # generate(out, stim, 4096*0.5, 4, 400, 0)

