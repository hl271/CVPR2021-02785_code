addpath('D:\Data\CVPR2021-02785\CVPR2021-02785\code\bin\');
addpath('D:\Download\eeglab2020_0\eeglab2020_0');

dir = 'D:\Data\CVPR2021-02785\CVPR2021-02785\data\imagenet40-1000-';
name = 'C:\Users\AI-INNOVATOR\AppData\Local\Temp\imagenet40-1000\imagenet40-1000-';
design_path = 'D:\Data\CVPR2021-02785\CVPR2021-02785\design\run';

eeglab;

band = 0;
notch = 0;

for subject = 1:1
    for run = 0:99
        bdf = sprintf('%s%d-%02d.bdf', dir, subject, run);
        out = sprintf('%s%d/', name, subject);
        stim = sprintf('%s-%02d.txt', design_path, run);
        trim = run==14;
        read_EEG(bdf, band, notch, 400, trim);
        generate(out, stim, 4096*0.5, 4, 400, 0);
    end
end

exit
