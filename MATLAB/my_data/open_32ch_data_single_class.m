% % 10s single class files
clear;clc;close all;
% DIR = 'S1\'; CLASS_LOC = 8; DECIM = true;
% DIR = 'S2\'; CLASS_LOC = 7; DECIM = true;
% DIR = 'S3\'; CLASS_LOC = 7; DECIM = false; 
DIR = 'S4\'; CLASS_LOC = 7; DECIM = false; 
% DIR = 'S5\'; CLASS_LOC = 7; DECIM = false; 
EXT = '.bdf';
addpath('C:\Users\Musa Mahmood\Dropbox (GaTech)\YeoLab\_SSVEP\_MATLAB-SSVEP-Classification\plugins\Biosig3.3.0\biosig\eeglab');
x = fileparts( which('sopen') );
rmpath(x);
addpath(x,'-begin'); PLOT = 0;
KEEP_ELECTRODES = 1:32; start = 1; whop = 32; wlen = 1024;
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
files = dir([DIR '*' EXT]);
w = 1; relevant_data_all = []; Y_all = [];
[b, a] = butter(3, 4.9*2/512, 'high');
OUTPUT_DIR = ['output_dir\psd\' DIR(1:end-1) '_psd_wlen', num2str(wlen) '\'];
for f = 1:length(files)
    filename = files(f).name
    class = filename(CLASS_LOC)
    % OLD COMMAND: EEG = pop_readbdf([DIR, filename], [], [], 16); % Oz is reference
	EEG = pop_biosig([DIR, filename], 'ref', 37:38);
    [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'gui','off');
    data_orig = double(EEG.data(KEEP_ELECTRODES, :))'; % Remove EXG electrodes
    Fs = EEG.srate; 
    
    for ch = KEEP_ELECTRODES
        if DECIM
            data(:, ch) = decimate(data_orig(:, ch), 2);
        else
            data(:, ch) = data_orig(:, ch);
        end
        ch_labels{ch} = EEG.chanlocs(ch).labels;
    end
    if DECIM
        Fs = Fs/2; % IF DECIMATING DATA!
    end
    wStart = start:whop:(length(data)-wlen); wEnd = wStart + wlen - 1;
    P = zeros(length(wStart), length(KEEP_ELECTRODES), wlen/2);
    Y_file = zeros(length(wStart), 1);
    for w = 1:length(wStart)
        selected_window = filtfilt(b, a, data(wStart(w):wEnd(w), :));
        Y_file(w) = str2double(class);
        for ch = KEEP_ELECTRODES
            [P(w, ch, :), F] = welch_estimator_ORIG(selected_window(:,ch), Fs, hann(wlen)); %pass unfiltered
            P(w, ch, :) = rescale_minmax(P(w, ch, :)); % rescale on a per-channel basis
        end
        sample = reshape(P(w, :, :), [size(P,2), size(P,3)])';
%         sample_conv = conv(sample(:,15), sample(:,17));
        if (PLOT)
            figure(2); plot_imagesc(F, sample);
%             subplot(1,2,2); plot_imagesc(F, sample_conv(1:length(F))); xlim([0 1])
            figure(3);
            subplot(1,2,1); plot(F, sample);
            rgb = input('Continue? \n');
        end
    end
    relevant_data_all = [relevant_data_all; P];
    Y_all = [Y_all; Y_file];
    clear data;
end
%%% Split into train and test samples (last 10% = test)
num_samples = size(Y_all, 1);
train_samples = floor(num_samples * 0.77);
relevant_data = relevant_data_all(1:train_samples, :, :);
Y = Y_all(1:train_samples, :);
fname_new = [OUTPUT_DIR, 'psd_w', num2str(wlen), '.mat'];
mkdir(OUTPUT_DIR);
save(fname_new, 'ch_labels', 'Fs', 'relevant_data', 'Y');
clear relevant_data Y
relevant_data = relevant_data_all(train_samples+1:end, :, :);
Y = Y_all(train_samples+1:end, :, :);
fname_new = [OUTPUT_DIR 'v\', 'psd_w', num2str(wlen), '.mat'];
mkdir([OUTPUT_DIR 'v\']);
save(fname_new, 'ch_labels', 'Fs', 'relevant_data', 'Y');



function [] = plot_imagesc(F, P)
    imagesc(1:2, F, P); 
    set(gca,'YDir','normal'); xlabel('Ch, #');ylabel('Frequency, Hz'); colormap(jet); cb = colorbar; ylabel(cb, 'Power (db)');
end