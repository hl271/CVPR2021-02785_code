function generate(save_root, txt, samples, down_ratio, n, video)
global data;
global trigger;
data = data(1:96, :)';
channel = 96;
EEG = zeros(n, samples, channel);
for i = 1:length(trigger)
    EEG(i, :, :) = reshape(data(trigger(i):trigger(i)+samples-1, :), [1, samples, channel]);
end
EEG = reshape(EEG, [n*samples, channel]);
EEG = reshape(EEG, [n, samples, channel]);
fid = fopen(txt);
tline = fgetl(fid);
i = 1;
while ischar(tline)
    if video==0
        tline = tline(1:end-5);
    end
    tline = [tline, '.mat'];
    eeg = reshape(EEG(i, :, :), [samples, channel]);
    eeg = eeg(1:down_ratio:end, :)';
    xform = zeros(96, 257);
    for c = 1:96
        sig = eeg(c, :);
        [Pxx, f] = pmtm(sig, 3, 512,  1024);
        xform(c, :) = Pxx;
    end
    eeg = xform;
    name = [save_root, tline];
    save(name, 'eeg');
    tline = fgetl(fid);
    i = i+1;
end
