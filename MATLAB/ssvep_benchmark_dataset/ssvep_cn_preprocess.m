%% SSVEP CN DATASET:
clear; close all; clc;
d = dir([pwd '\S*.mat']);
output_dir = 'output_dir\no_preprocessing\';
mkdir(output_dir);
% filename = 'S1.mat';load(filename);  %temp: 
Fs = 250;
epochs = 6;
Y = [0, 1, 2, 3, 4]; % 0 = null, 1-4 = ssvep. 
select_freqs = [1, 3, 5, 8];
select_chs = 1:64;
relevant_data = zeros(1500, length(select_chs)); 
[b, a] = butter(3, 5*2/Fs, 'high');
% [res, b, a] = customFilt(zeros(32,1), 250, [3 100], 5);
for f = 1:length(d)
    filename = d(f).name; load(filename);
    data_select = data(select_chs, :, select_freqs, :);
    sz_select = size(data_select);
    for ep = 1:sz_select(4)
        for cl = 1:sz_select(3)
            single_epoch = data_select(:, :, cl, ep)';
            for ch = 1:length(select_chs)
                relevant_data(:, ch) = (filtfilt(b, a, single_epoch(:, ch)));
            end
            relevant_data = rescale_minmax(relevant_data);
            % first 125 pts are class 0:
            relevant_data(:, length(select_chs)+1) = 0; %(2) = 1:250
            relevant_data(126:end, length(select_chs)+1) = cl;
            mkdir([output_dir, filename(1:end-4)]);
            f_n = [output_dir, filename(1:end-4), '\epoch', num2str(ep), ...
                '_class', num2str(cl), '.mat'];
            save(f_n, 'relevant_data');
        end
    end
end
