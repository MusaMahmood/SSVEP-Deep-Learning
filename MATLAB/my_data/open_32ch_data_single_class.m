% % 10s single class files
clear;clc;close all;
DIR = 'S1\'; CLASS_LOC = 8;
% DIR = 'S2\'; CLASS_LOC = 7;
OUTPUT_DIR = [DIR(1:end-1) '_decimate\'];
EXT = '.bdf';
addpath('C:\Users\Musa Mahmood\Dropbox (GaTech)\YeoLab\_SSVEP\_MATLAB-SSVEP-Classification\plugins\Biosig3.3.0\biosig\eeglab');
x = fileparts( which('sopen') );
rmpath(x);
addpath(x,'-begin');
KEEP_ELECTRODES = 1:40;
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
files = dir([DIR '*' EXT]);
for f = 1:length(files)
    filename = files(f).name
    class = filename(CLASS_LOC);
    % OLD COMMAND: EEG = pop_readbdf([DIR, filename], [], [], 16); % Oz is reference
	EEG = pop_biosig([DIR, filename], 'ref', 16);
    [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'gui','off');
    data = double(EEG.data(KEEP_ELECTRODES, :))'; % Remove EXG electrodes
    Fs = EEG.srate; Fs2 = Fs/2;
%     [relevant_data, Fs] = dsample(data, Fs);% Downsample
    % ADD CLASS TO END:
    for i = KEEP_ELECTRODES
        relevant_data(:, i) = decimate(data(:, i), 2); 
        ch_labels{i} = EEG.chanlocs(i).labels;
%         [P2(i, :), F2] = welch_psd(rescale_minmax(data(end/2+1:end,i)), 512, hann(512 * 4));
%         [P(i, :), F] = welch_psd(rescale_minmax(relevant_data(end/2+1:end,i)), Fs2, hann(Fs2 * 4));    
    end
    relevant_data(:, end+1) = str2double(class);
%     figure(9); plot(F, P); xlim([5 45]);
%     figure(10); plot(F2, P2); xlim([5 45]);
%     figure(2); imagesc(KEEP_ELECTRODES, F(51:end), (P(:,51:end))');ylim([0 40]); set(gca,'YDir','normal'); xlabel('Ch, #');ylabel('Frequency, Hz'); colormap(jet); cb = colorbar; ylabel(cb, 'Power (db)')
    fname_new = [OUTPUT_DIR, filename(1:end-4), '.mat'];
    mkdir(OUTPUT_DIR);
    save(fname_new, 'ch_labels', 'Fs', 'relevant_data');
    clear relevant_data;
end

function [Y, Fs] = dsample(X, Fs_orig)
    if mod(length(X), 2) ~= 0
        X = X(1:end-1, :);
    end
    Y = zeros(length(X)/2, size(X, 2));
    for j = 1: size(X, 2)
        for i = 1:length(X)/2
            Y(i, j) = (X(2*i-1, j) + X(2*i, j))/2; % arithmetic mean
        end
    end
    Fs = Fs_orig/2;
end

function [] = disp_stft(X, Fs, winLim)
    wlen = Fs;
    nfft = 2*Fs;
    h = 32;
    [S, F, T] = stft(X, wlen, h, nfft, Fs);
    select = F<winLim(2) & F>winLim(1);
    F1 = F(select);
    S1 = 20*log10( abs( S(select,:) ) /wlen/K + 1E-6 );
    imagesc(T,F1,S1),ylim(winLim),xlim([min(T),max(T)]);set(gca,'YDir','normal');colorbar;colormap(jet);
end