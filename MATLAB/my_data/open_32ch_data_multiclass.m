% % 10s single class files
clear;clc;close all;
DIR = 'S1_32CH\';
EXT = '.bdf';
addpath('C:\Users\mmahmood31\Dropbox (GaTech)\YeoLab\_SSVEP\_MATLAB-SSVEP-Classification\plugins\Biosig3.3.0\biosig\eeglab');
x = fileparts( which('sopen') );
   rmpath(x);
   addpath(x,'-begin');
KEEP_ELECTRODES = 1:40;
files = dir([DIR '*' EXT]);
[~, b, a] = customFilt(zeros(500, 1), 512, [2 50], 3);
for f = 1:length(files)
    filename = files(f).name;
    EEG = pop_readbdf([DIR, filename], [], [], 32); % Oz is reference
%     data = filtfilt(b, a, double(EEG.data(KEEP_ELECTRODES, :))');% Remove EXG electrodes
    data = double(EEG.data(KEEP_ELECTRODES, :));
%     data_rescaled = rescale_minmax(data);
    Fs = EEG.srate;
    [relevant_data, Fs] = dsample(data, Fs);% Downsample
    % ADD CLASS TO END:
    for i = KEEP_ELECTRODES
        EEG.chanlocs(i).labels
        ch_labels{i} = EEG.chanlocs(i).labels;
%         figure(1); disp_stft(relevant_data(:,i), Fs, [0 Fs/2]);
%         figure(2); [P, F] = welch_psd(relevant_data(:,i), Fs, hann(Fs*2));
%         plot(F, P);
%         input('Continue: \n'); commandwindow;
    end
    DIR2 = [DIR(1:end-1) '_ds/']
    fname_new = [DIR2, filename(1:end-4), '_RAW_downsampledOnly.mat'];
    mkdir(DIR2);
    save(fname_new, 'ch_labels', 'Fs', 'relevant_data');
    clear relevant_data;
end

% 
% [b, a] = butter(3, 5*2/Fs, 'high');
% RECORD_Filt = filtfilt(b, a, relevant_data);

function [Y, Fs] = dsample(X, Fs_orig)
    if mod(length(X), 2) ~= 0
        X = X(1:end-1, :);
    end
%     prealloc Y:
    Y = zeros(length(X)/2, size(X, 2));
    for j = 1: size(X, 2)
        for i = 1:length(X)/2
            Y(i, j) = (X(2*i-1, j) + X(2*i, j))/2; % arithmetic mean
        end
    end
    Fs = Fs_orig/2;
end

function [] = disp_stft(X, Fs, winLim)
    wlen = 5*Fs;
    nfft = 5*Fs;
    h = 32;
    K = sum(hammPeriodic(wlen))/wlen;
    [S, F, T] = stft(X, wlen, h, nfft, Fs);
    select = F<winLim(2) & F>winLim(1);
    F1 = F(select);
    S1 = 20*log10( abs( S(select,:) ) /wlen/K + 1E-6 );
    imagesc(T,F1,S1),ylim(winLim),xlim([min(T),max(T)]);set(gca,'YDir','normal');colorbar;colormap(jet);
end