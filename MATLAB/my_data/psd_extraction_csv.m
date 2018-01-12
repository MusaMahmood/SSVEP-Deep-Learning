%% PSD Extraction:
clear; close all; clc;
% d = dir([pwd '\S*.csv']);
Subject = 'S0_2ch\';
d = dir([Subject 'S*.csv']);
output_dir = ['output_dir\psd\' Subject];
mkdir(output_dir); PLOT = 1;
Fs = 250;
select_chs = 1:2;
start = 1; whop = 32; wlen = 512;
for f = 1:length(d)
    filename = d(f).name; 
    data = csvread([Subject filename]);
    wStart = start:whop:(length(data)-wlen); wEnd = wStart + wlen - 1;
    P = zeros(length(wStart), length(select_chs), wlen/2);
    relevant_data = P;
    Y = zeros(length(wStart), 1);
    for w = 1:length(wStart)
        selected_window = data(wStart(w):wEnd(w), :);
        if sum(selected_window(:, 3) == selected_window(1, 3)) == wlen
            CLASS = selected_window(1, 3)
            Y(w) = selected_window(1, 3);
            if wlen == 256
                temp_4 = tf_psd_rescale_w256(selected_window(:,select_chs));
            elseif wlen == 384
                temp_4 = tf_psd_rescale_w384(selected_window(:,select_chs));
            elseif wlen == 512
                temp_4 = tf_psd_rescale_w512(selected_window(:,select_chs));
            end
            for ch = 1:length(select_chs)
                [P(w, ch, :), F] = welch_estimator_ORIG(selected_window(:,ch), Fs, hann(wlen)); %pass unfiltered
                P(w, ch, :) = rescale_minmax(P(w, ch, :)); % rescale on a per-channel basis
            end
            relevant_data(w, :, :) = P(w, :, :); 
            sample = reshape(P(w, :, :), [size(P,2), size(P,3)])';
            sample_conv = conv(sample(:,1), sample(:,2));
            if (PLOT)
                figure(1); subplot(1,2,1); plot_imagesc(F, sample);
                subplot(1,2,2); plot_imagesc(F, sample_conv(1:length(F))); xlim([0 1])
                figure(2);
                subplot(1,2,1); plot(F, sample);
                subplot(1,2,2); plot(F, sample_conv(1:length(F)));
                rgb = input('Continue? \n');
            end
        end
    end         
%     mkdir([output_dir]);
%     f_n = [output_dir, filename(1:end-4), '_nofilt_psd_wlen_' num2str(wlen) '.mat'];
%     save(f_n, 'relevant_data', 'Y');
    clear Y
end

function [] = plot_imagesc(F, P)
    imagesc(1:2, F, P); 
    set(gca,'YDir','normal'); xlabel('Ch, #');ylabel('Frequency, Hz'); colormap(jet); cb = colorbar; ylabel(cb, 'Power (db)');
end