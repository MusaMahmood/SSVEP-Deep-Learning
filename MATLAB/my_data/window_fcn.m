% Data Analysis; SSVEP:
clear; close all; clc;
Subject = 'S0_2ch\';
% Subject = '2018-01-05-rob\';
% Subject = '2018-01-05-SJ\';
d = dir([Subject 'S*.csv']);
PLOT = 1;
Fs = 250;
select_chs = 1:2;
start = 1; whop = 32; wlen = 256;
[b, a] = butter(3, 5*2/Fs, 'high');
for f = 1:length(d)
    filename = d(f).name; 
    data = csvread([Subject filename]);
    fdata = filtfilt(b, a, data(:, 1:2));
    wStart = start:whop:(length(data)-wlen); wEnd = wStart + wlen - 1;
%     P = zeros(length(wStart), length(select_chs), wlen/2);
%     relevant_data = P;
%     Y = zeros(length(wStart), 1);
    if(PLOT) 
        figure(1);
        subplot(2,1,1);
        plot(fdata(250:250*5+200,1)); title('Channel 1');
        subplot(2,1,2);
        plot(fdata(250:250*5+200,2)); title('Channel 2');
        figure(2);
        subplot(2,1,1);
        plot(fdata(18*250:250*22,1)); title('Channel 1');
        subplot(2,1,2);
        plot(fdata(18*250:250*22,2)); title('Channel 2');
    end
%     for w = 1:length(wStart)
%         selected_window = data(wStart(w):wEnd(w), :);
%         if sum(selected_window(:, 3) == selected_window(1, 3)) == wlen
%             CLASS = selected_window(1, 3)
%             Y(w) = selected_window(1, 3);
%             for ch = 1:length(select_chs)
%                 [P(w, ch, :), F] = welch_estimator_ORIG(selected_window(:,ch), Fs, hann(wlen)); %pass unfiltered
%                 P(w, ch, :) = rescale_minmax(P(w, ch, :)); % rescale on a per-channel basis
%             end
%             relevant_data(w, :, :) = P(w, :, :);
%             if (PLOT)
%                 imagesc(select_chs, F, reshape(relevant_data(w, :, :), [size(P,2), size(P,3)])'); %ylim([5 40])
%                 set(gca,'YDir','normal'); xlabel('Ch, #');ylabel('Frequency, Hz'); colormap(jet); cb = colorbar; ylabel(cb, 'Power (db)')
%                 rgb = input('Continue? \n');
%             end
%         end
%     end         
%     mkdir([output_dir]);
%     f_n = [output_dir, filename(1:end-4), '_nofilt_psd_wlen_' num2str(wlen) '.mat'];
%     save(f_n, 'relevant_data', 'Y');
%     clear Y
end

%%
[DATA, filename] = csvread('2018-01-05-rob\S2_T1.1.csv');
%save new data: (DO ONLY ONCE!!!):
Fs = 250;
rS = 0*Fs; %Remove From Start
rE = 0*Fs; %Remove From End
F = [6.5 43];
winLim = [1 40];
numch = 2; datach = DATA(rS+1:end-rE,1:numch);
% Plot Raw Data
figure(95); 
% sel = 1:(10*Fs);
% subplot(2,1,1);
plot(DATA(:,1));
% subplot(2,1,2); plot(DATA(sel,2));
%%-Plot Analysis: %{
filtch = zeros(size(datach,1),numch);
hannWin = hann(2048); wlen = 512; h=32; nfft = 4096; K = sum(hamming(wlen, 'periodic'))/wlen;
for i = 1:numch %    
	filtch(:,i) = customFilt(datach(:,i),Fs,F,3); %figure(1); hold on; plot(filtch(:,i));
%     [f, P1] = get_fft_data(filtch(:,i),Fs); figure(2);hold on; plot(f,P1),xlim(winLim);
end
hannWin2 = hann(4096);
% filename = strrep(filename,'_','-');
fH = figure(4);
set(fH, 'Position', [15, 15, 1600, 920]); %Spect
for i = 1:numch
    [S1,wfreqs] = welch_psd(filtch(:,i), Fs, hannWin);
    M(i) = max(S1);
%     [S1_1{i},~] = welch_psd(DATA(:,i), Fs, hannWin);
    hold on; subplot(2,2,[1 2]);  plot(wfreqs, S1),xlim(winLim);xlabel('Frequency (Hz)'),ylabel('Power Density (W/Hz)')%,title([ filename(end-20:end), ' - ',  'Power Spectral Density Estimate']);
end
M2 = max(M); 
legend('Channel 1','Channel 2');
for i = 1:2
    subplot(2,2,i+2) % filtch
    [S1, f1, t1] = stft( filtch(:,i), wlen, h, nfft, Fs ); S2 = 20*log10(abs(S1(f1<winLim(2) & f1>winLim(1),:))/wlen/K + 1e-6); 
    imagesc(t1,f1(f1<winLim(2) & f1>winLim(1)),S2),xlim([min(t1) max(t1)]),ylim(winLim);
    set(gca,'YDir','normal');xlabel('Time, s');ylabel('Frequency, Hz');colormap(jet)
    cb = colorbar;ylabel(cb, 'Power (db)')
    title(['Ch' num2str(i)]);
end