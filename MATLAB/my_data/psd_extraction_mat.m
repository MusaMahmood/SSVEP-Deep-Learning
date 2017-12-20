%% PSD Extraction:
clear; clc; figure(1), figure(2); clf(1); clf(2);
DIR = 'S2_mat\';
d = dir([DIR, 't*.mat']);
output_dir = [DIR, 'output_psd\'];
mkdir(output_dir); PLOT = 1;
Fs = 256;
select_chs = 1:40;
[b, a] = butter(3, 5*2/Fs, 'high');
start = 1; whop = 32; wlen = 512;
for f = 1:length(d)
    filename = d(f).name
    load([DIR filename]);
    data = relevant_data;
    wStart = start:whop:(length(data)-wlen); wEnd = wStart + wlen - 1;
    P = zeros(length(wStart), length(select_chs), wlen/2);
    relevant_data = P;
    Y = zeros(length(wStart), 1);
    for w = 1:length(wStart)
       	fprintf('[ %d to %d of %d ] \n' , wStart(w), wEnd(w), wEnd(end));
        selected_window = data(wStart(w):wEnd(w), :);
        if sum(selected_window(:, end) == selected_window(1, end)) == wlen
            Y(w) = selected_window(1, end);
            filtered_window = filtfilt(b, a, selected_window);
            for ch = 1:length(select_chs)
                [P(w, ch, :), F] = welch_psd(filtered_window(:,ch), Fs, hann(wlen));
            end
            relevant_data(w, :, :) = rescale_minmax(P(w, :, :));
            if (PLOT)
                figure(1);
                plot_psd_data(F, relevant_data(w, :, :), [1:32]);
                figure(2);
                imagesc(select_chs, F, reshape(relevant_data(w, :, :), [size(P,2), size(P,3)])'); ylim([1 40])
                set(gca,'YDir','normal'); xlabel('Ch, #');ylabel('Frequency, Hz'); colormap(jet); cb = colorbar; ylabel(cb, 'Power (db)')
                input('Continue? \n'); commandwindow;
            end
        end
    end         
    mkdir([output_dir]);
    f_n = [output_dir, filename(1:end-4), '_psd.mat'];
    save(f_n, 'relevant_data', 'Y');
    clear Y relevant_data
end


function [] = plot_psd_data(F, P, chs)
    if length(size(P)) > 2
        P = reshape(P, [size(P,2), size(P,3)]);
    end
    plot(F, P(chs, :));
end