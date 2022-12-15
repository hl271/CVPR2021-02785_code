% Run this line of script (on Git bash) to load bdf file using pop_biosig
% matlab -nosplash -nodesktop -r "run('D:\Data\CVPR2021-02785\CVPR2021-02785\code\bin\test_EEG.m');"
addpath('D:\Data\CVPR2021-02785\CVPR2021-02785\code\bin\');
addpath('D:\Download\eeglab2020_0\eeglab2020_0');

src_path = 'D:\Data\CVPR2021-02785\CVPR2021-02785\data\imagenet40-1000-';
temp_path = 'C:\Users\AI-INNOVATOR\AppData\Local\Temp\imagenet40-1000\imagenet40-1000-';
design_path = 'D:\Data\CVPR2021-02785\CVPR2021-02785\design\run';

eeglab;

band = 0;
notch = 0;

sampling_rate = 1024;
down_factor = 1;

for subject = 1:1
    for run = 0:0
        bdf = sprintf('%s%d-%02d.bdf', src_path, subject, run);
        out = sprintf('%s%d/', temp_path, subject);
        stim = sprintf('%s-%02d.txt', design_path, run);
        trim = run==14;
        read_EEG(bdf, band, notch, 400, trim);
        generate(out, stim, sampling_rate*0.5, down_factor, 400, 0);
    end
end

exit



