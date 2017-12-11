%% PSD Extraction:
clear; close all; clc;
d = dir([pwd '\S*.csv']);
output_dir = 'output_dir\psd\';
mkdir(output_dir); PLOT = 0;
Fs = 250;
select_chs = 1:2;
[b, a] = butter(3, 5*2/Fs, 'high');
start = 125; whop = 32; wlen = 256;
for f = 1:length(d)
    filename = d(f).name; 
    data = csvread(filename);
    wStart = start:whop:(length(data)-wlen); wEnd = wStart + wlen - 1;
    P = zeros(length(wStart), length(select_chs), wlen/2);
    relevant_data = P;
    Y = zeros(length(wStart), 1);
    for w = 1:length(wStart)
        selected_window = data(wStart(w):wEnd(w), :);
        if sum(selected_window(:, 3) == selected_window(1, 3)) == wlen
            Y(w) = selected_window(1, 3);
            filtered_window = filtfilt(b, a, selected_window);
            for ch = 1:length(select_chs)
                [P(w, ch, :), F] = welch_estimator_ORIG(filtered_window(:,ch), Fs, hann(wlen));
            end
            relevant_data(w, :, :) = rescale_minmax(P(w, :, :));
            if (PLOT)
                imagesc(select_chs, F, reshape(relevant_data(w, :, :), [size(P,2), size(P,3)])'); ylim([5 40])
                set(gca,'YDir','normal'); xlabel('Ch, #');ylabel('Frequency, Hz'); colormap(jet); cb = colorbar; ylabel(cb, 'Power (db)')
            end
        end
    end         
    mkdir([output_dir]);
    f_n = [output_dir, filename(1:end-4), '_psd.mat'];
    save(f_n, 'relevant_data', 'Y');
    clear Y
end