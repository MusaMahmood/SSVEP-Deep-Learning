%% SSVEP CN DATASET:
clear; close all; clc;
d = dir([pwd '\S*.mat']);
output_dir = 'output_dir\f3c\';
mkdir(output_dir);
Fs = 250;
epochs = 6;
select_freqs = [15 28 37]; num_classes = length(select_freqs); %[1, 3, 5, 8, 14];
select_chs = 1:64; num_chs = length(select_chs);
relevant_data_0 = zeros(1500, num_chs); 
relevant_data = [];
[b, a] = butter(3, 5*2/Fs, 'high');
for f = 1
    filename = d(f).name; load(filename);
    data_select = data(select_chs, :, select_freqs, :);
    sz_select = size(data_select);
    for ep = 1:sz_select(4)
        for cl = 1:sz_select(3)
            single_epoch = data_select(:, :, cl, ep)';
            for ch = 1:length(select_chs)
                relevant_data_0(:, ch) = (filtfilt(b, a, single_epoch(:, ch)));
            end
            relevant_data_0 = rescale_minmax(relevant_data_0);
            relevant_data_1 = relevant_data_0(126:end, select_chs);% Cut first 125 pts (null class):
            relevant_data_1(:, num_chs+1) = cl;% label:
            relevant_data = [relevant_data; relevant_data_1];
        end
        mkdir([output_dir, filename(1:end-4)]);
        f_n = [output_dir, filename(1:end-4), '\epoch', num2str(ep) '.mat'];
        save(f_n, 'relevant_data');
        relevant_data = [];
    end
end
%{

%}