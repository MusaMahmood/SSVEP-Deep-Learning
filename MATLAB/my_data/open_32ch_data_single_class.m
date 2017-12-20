% % 10s single class files
clear;clc;close all;
DIR = 'S2\';
OUTPUT_DIR = [DIR(1:end-1) '_mat\'];
EXT = '.bdf';
addpath('C:\Users\Musa Mahmood\Dropbox (GaTech)\YeoLab\_SSVEP\_MATLAB-SSVEP-Classification\plugins\Biosig3.3.0\biosig\eeglab');
x = fileparts( which('sopen') );
rmpath(x);
addpath(x,'-begin');
KEEP_ELECTRODES = 1:40;

files = dir([DIR '*' EXT]);
for f = 1:length(files)
    filename = files(f).name
    class = filename(7);
    [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
	EEG = pop_biosig([DIR, filename], 'ref', 16);
    [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'gui','off');
%     EEG = pop_readbdf([DIR, filename], [], [], 16); % Oz is reference
    data = double(EEG.data(KEEP_ELECTRODES, :))'; % Remove EXG electrodes
    Fs = EEG.srate;
    [relevant_data, Fs] = dsample(data, Fs);% Downsample % relevant_data = downsample(data, 2);
    relevant_data(:, end+1) = str2double(class);
    % ADD CLASS TO END:
    for i = KEEP_ELECTRODES
        ch_labels{i} = EEG.chanlocs(i).labels;
%         [P(i, :), F] = welch_psd(rescale_minmax(relevant_data(:,i)), Fs, hann(Fs*4));
    end
%     figure(2); imagesc(KEEP_ELECTRODES, F(51:end), (P(:,51:end))');ylim([0 40]); set(gca,'YDir','normal'); xlabel('Ch, #');ylabel('Frequency, Hz'); colormap(jet); cb = colorbar; ylabel(cb, 'Power (db)')
%     figure(3); plot(F, P); xlim([6 45]); input('Continue: \n'); commandwindow;
    fname_new = [OUTPUT_DIR, filename(1:end-4), '_RAW_downsampled.mat'];
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