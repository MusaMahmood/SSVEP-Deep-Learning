%% PSD Extraction:
clear; close all; clc;
% d = dir([pwd '\S*.csv']);
Subject = 'S0_2ch\';
% Subject = '2018-01-05-rob\';
d = dir([Subject 'S*.csv']);
output_dir = ['output_dir\psd\' Subject];
mkdir(output_dir); PLOT = 0;
Fs = 250; h = 1/Fs;
select_chs = 1:2;
start = 1; whop = 32; wlen = 256;
[b, a] = butter(3, 1.33*2/Fs, 'high');
for f = 1%:length(d)
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
            sample = filtfilt(b,a,selected_window(:,select_chs));
            if wlen == 256
                temp_4 = tf_psd_rescale_w256(sample);
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
             %reshape(P(w, :, :), [size(P,2), size(P,3)])';
%             sample_conv = conv(sample(:,1), sample(:,2));
            if (PLOT && CLASS == 2)
%                 figure(1); subplot(1,2,1); plot_imagesc(F, sample);
%                 subplot(1,2,2); plot_imagesc(F, sample_conv(1:length(F))); xlim([0 1])
%                 figure(2);
%                 subplot(1,2,1); plot(F, sample);
%                 subplot(1,2,2); plot(F, sample_conv(1:length(F)));
                plot(0:h:256*h-h,sample); xlim([0, 1.0]); ylim([-2.5E-4, 2.5E-4])
                xlabel('Time, s'); ylabel('V'); figure;
                plot_imagesc(temp_4);
                %plot(F, temp_4); xlim([5, 40]);
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
    if nargin > 1
        imagesc(1:2, F, P); 
    else
        R = [0,0.976562500000000,1.95312500000000,2.92968750000000,3.90625000000000,4.88281250000000,5.85937500000000,6.83593750000000,7.81250000000000,8.78906250000000,9.76562500000000,10.7421875000000,11.7187500000000,12.6953125000000,13.6718750000000,14.6484375000000,15.6250000000000,16.6015625000000,17.5781250000000,18.5546875000000,19.5312500000000,20.5078125000000,21.4843750000000,22.4609375000000,23.4375000000000,24.4140625000000,25.3906250000000,26.3671875000000,27.3437500000000,28.3203125000000,29.2968750000000,30.2734375000000,31.2500000000000,32.2265625000000,33.2031250000000,34.1796875000000,35.1562500000000,36.1328125000000,37.1093750000000,38.0859375000000,39.0625000000000,40.0390625000000,41.0156250000000,41.9921875000000,42.9687500000000,43.9453125000000,44.9218750000000,45.8984375000000,46.8750000000000,47.8515625000000,48.8281250000000,49.8046875000000,50.7812500000000,51.7578125000000,52.7343750000000,53.7109375000000,54.6875000000000,55.6640625000000,56.6406250000000,57.6171875000000,58.5937500000000,59.5703125000000,60.5468750000000,61.5234375000000,62.5000000000000,63.4765625000000,64.4531250000000,65.4296875000000,66.4062500000000,67.3828125000000,68.3593750000000,69.3359375000000,70.3125000000000,71.2890625000000,72.2656250000000,73.2421875000000,74.2187500000000,75.1953125000000,76.1718750000000,77.1484375000000,78.1250000000000,79.1015625000000,80.0781250000000,81.0546875000000,82.0312500000000,83.0078125000000,83.9843750000000,84.9609375000000,85.9375000000000,86.9140625000000,87.8906250000000,88.8671875000000,89.8437500000000,90.8203125000000,91.7968750000000,92.7734375000000,93.7500000000000,94.7265625000000,95.7031250000000,96.6796875000000,97.6562500000000,98.6328125000000,99.6093750000000,100.585937500000,101.562500000000,102.539062500000,103.515625000000,104.492187500000,105.468750000000,106.445312500000,107.421875000000,108.398437500000,109.375000000000,110.351562500000,111.328125000000,112.304687500000,113.281250000000,114.257812500000,115.234375000000,116.210937500000,117.187500000000,118.164062500000,119.140625000000,120.117187500000,121.093750000000,122.070312500000,123.046875000000,124.023437500000];
        imagesc(R, 1:2, F);
    end
    set(gca,'YDir','normal'); ylabel('Ch, #'); xlabel('Frequency, Hz'); colormap(jet); %cb = colorbar; ylabel(cb, 'Normalized Power Density');
end