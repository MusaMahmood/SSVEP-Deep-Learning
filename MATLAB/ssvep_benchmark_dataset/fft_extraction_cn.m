%% FFT Extraction:
clear; close all; clc;
d = dir([pwd '\S*.mat']);
output_dir = 'output_dir\fft\';
mkdir(output_dir); PLOT = 0;
% filename = 'S1.mat';load(filename);  %temp: 
Fs = 250;
epochs = 6;
Y_classes = [0, 1, 2, 3, 4]; % 0 = null, 1-4 = ssvep. 
select_freqs = [1, 3, 5, 8, 13];
select_chs = 1:64;
% relevant_data = zeros(1500, length(select_chs));
[b, a] = butter(3, 5*2/Fs, 'high');
start = 125; whop = 32; wlen = 256;
wStart = start:whop:(1500-wlen);
wEnd = wStart + wlen - 1;
F = Fs*(0:(wlen/2))/wlen;
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
                    P2 = fft(filtered_window(:,ch), wlen);
                    P2 = abs(P2/wlen);
                    P1 = P2(1:wlen/2+1);
                    P1(2:end-1) = 2*P1(2:end-1);
                    P(w, ch, :) = P1(1:end-1);
                end
                relevant_data(w, :, :) = rescale_minmax(P(w, :, :));
                if (PLOT)
                    plot_image(relevant_data, w, select_chs, F);
                end
                Y(w) = cl;
            end         
            % first 125 pts are class 0:
            mkdir([output_dir, filename(1:end-4)]);
            f_n = [output_dir, filename(1:end-4), '\epoch', num2str(ep), ...
                '_class', num2str(cl), '.mat'];
            save(f_n, 'relevant_data', 'Y');
        end
    end
end

function N = plot_image(Pimage, w, select_chs, F)
    imagesc(select_chs, F, reshape(Pimage(w, :, :), [size(Pimage,2), size(Pimage,3)])'); ylim([5 40])
    set(gca,'YDir','normal'); xlabel('Ch, #');ylabel('Frequency, Hz'); colormap(jet); cb = colorbar; ylabel(cb, 'Power (db)')
    N = 0;
end
