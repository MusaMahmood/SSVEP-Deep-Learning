%% FFT Extraction:
clear; close all; clc;
d = dir([pwd '\S*.csv']);
output_dir = 'output_dir\windowed_400\';
mkdir(output_dir); PLOT = 0;
Fs = 250;
epochs = 6;
select_chs = 1:2;
[b, a] = butter(3, 5*2/Fs, 'high');
start = 250; whop = 32; wlen = 400;
for f = 1:length(d)
    filename = d(f).name; data = csvread(filename);
    single_epoch = data(:, select_chs);
    classes = data(:, 1 + max(select_chs));
    wStart = start:whop:(length(data)-wlen);
    wEnd = wStart + wlen - 1;
    w = 1;
    for i = 1:length(wStart)
        filtered_window = filtfilt(b, a, single_epoch(wStart(i):wEnd(i), :));
        class_window = classes(wStart(i):wEnd(i), :);
        for ch = 1:length(select_chs)
            rescaled_data(i, ch, :) = rescale_minmax(filtered_window(:, ch));
        end
        if (PLOT)
            hold on;
            plot(reshape(rescaled_data(i, 1:ch, :), [256, ch]));
            in1 = input('Continue? \n'); 
            hold off;
        end
        if sum(class_window == class_window(1)) == wlen
            relevant_data(w, ch, :) = rescaled_data(i, ch, :);
            Y(w) = class_window(1);
            w = w+1;
        end
    end         
    % first 125 pts are class 0:
%     mkdir([output_dir, filename(1:end-4)]);
    f_n = [output_dir, filename(1:end-4), '.mat'];
    save(f_n, 'relevant_data', 'Y');
    clear relevant_data Y
end

