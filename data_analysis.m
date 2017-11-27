%% MAMEM DATA ANALYSIS:
clear;clc;close all;
SELECT_CHANNELS = [1:256];
% SELECT_CHANNELS = [114, 116, 123, 126, 137, 150, 158, 168];
Fs = 250;
dr = dir('DATA\output_csv\S011\*.mat');
plot_data = [];
select = 1:1250;
hannWin = hann(1024); wlen = 256; h=64; nfft = 2048;
K = sum(hamming(wlen, 'periodic'))/wlen; winLim = [0, 45];
X = 1:Fs;
[~,b,a] = customFilt(X,Fs,[1, 120],3);
i = 1;
for i = 12 %1:length(dr)
    load([dr(i).folder, '\', dr(i).name])
    select_data = relevant_data(:,SELECT_CHANNELS);
    % Filter data:
    CLASS = relevant_data(1, 257)
    for ch = 1:length(SELECT_CHANNELS)
        % Filter:
        f_data(:,ch) = filtfilt(b,a,select_data(select,ch));
        %scale from uV to V:
%         f_data(:,ch) = (f_data(:,ch)/1000000);
        %rescale to 1
%         l = -1; u = 1;
%         minv = min(f_data(:,ch));
%         maxv = max(f_data(:,ch));
%         f_data(:,ch) = l + ((f_data(:,ch)-minv)./(maxv-minv)).*(u-l)
%         if(plot_data)
%             figure(7); hold on;
%             plot(f_data(:,ch));
%         end
        % PSD
        [S(:,ch), freq] = welch_psd(f_data(:,ch),Fs,hannWin);
        
        % STFT:
%         [S1, f1, t1] = stft( f_data(:,ch), wlen, h, nfft, Fs ); S2 = 20*log10(abs(S1(f1<winLim(2) & f1>winLim(1),:))/wlen/K + 1e-6);
%{
        if(isempty(plot_data))    
            figure(1); hold on;
%             subplot(2,2,[1,2]); 
            plot(freq,S(:,ch)), xlim(winLim); title([num2str(ch), ' channel']);
            
%             subplot(2,2,[3,4]); imagesc(t1,f1(f1<winLim(2) & f1>winLim(1)),S2),xlim([min(t1) max(t1)]),ylim(winLim);
%             set(gca,'YDir','normal');xlabel('Time, s');ylabel('Frequency, Hz');colormap(jet);
%             cb = colorbar;ylabel(cb, 'Power (db)')
        end
        
        if (mod(ch,16) == 0)
            plot_data = input('Continue? \n');
            clf(1)
        end
%}
    end
end

% plot(freq, S(:, 1)); xlim(winLim);
%% CLASS 1:
% plot(S(45,:));
%% CLASS 2:
plot(S(61,:));
