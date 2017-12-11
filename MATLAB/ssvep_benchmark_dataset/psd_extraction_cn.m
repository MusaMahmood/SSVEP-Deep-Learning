%% PSD Extraction:
clear; close all; clc;
d = dir([pwd '\S*.mat']);
output_dir = 'output_dir\psd\';
mkdir(output_dir); PLOT = 1;
% filename = 'S1.mat';load(filename);  %temp: 
Fs = 250;
epochs = 6;
Y = [0, 1, 2, 3, 4]; % 0 = null, 1-4 = ssvep. 
select_freqs = [1, 3, 5, 8];
select_chs = 1:64;
% relevant_data = zeros(1500, length(select_chs));
[b, a] = butter(3, 5*2/Fs, 'high');
start = 125; whop = 32; wlen = 256;
wStart = start:whop:(1500-wlen);
wEnd = wStart + wlen - 1;
P = zeros(length(wStart), length(select_chs), wlen/2);
relevant_data = P; 
for f = 1%:length(d)
    filename = d(f).name; load(filename);
    data_select = data(select_chs, :, select_freqs, :);
    sz_select = size(data_select);
    for ep = 1:sz_select(4)
        for cl = 1:sz_select(3)
            single_epoch = data_select(:, :, cl, ep)';
            for w = 1:length(wStart)
                filtered_window = filtfilt(b, a, single_epoch(wStart(w):wEnd(w), :));
                for ch = 1:length(select_chs)
                    [P(w, ch, :), F] = welch_estimator_ORIG(filtered_window(:,ch), Fs, hann(wlen));
                end
                relevant_data(w, :, :) = rescale_minmax(P(w, :, :));
                if (PLOT)
                    imagesc(select_chs, F, reshape(P(w, :, :), [size(P,2), size(P,3)])'); ylim([5 40])
                    set(gca,'YDir','normal'); xlabel('Ch, #');ylabel('Frequency, Hz'); colormap(jet); cb = colorbar; ylabel(cb, 'Power (db)')
                end
                Y(w) = cl;
            end         

            % first 125 pts are class 0:
%             relevant_data(:, length(select_chs)+1) = 0; %(2) = 1:250
%             relevant_data(126:end, length(select_chs)+1) = cl;
            mkdir([output_dir, filename(1:end-4)]);
            f_n = [output_dir, filename(1:end-4), '\epoch', num2str(ep), ...
                '_class', num2str(cl), '.mat'];
            save(f_n, 'relevant_data', 'Y');
        end
    end
end