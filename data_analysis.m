%% MAMEM DATA ANALYSIS:
clear;clc;close all;
SELECT_CHANNELS = [114, 116, 126, 150, 168];
Fs = 250;
dr = dir('DATA\output_csv\S001\*.mat');
plot_data = 1;
select = 1:512;
hannWin = hann(256); wlen = 256; h=64; nfft = 2048;
K = sum(hamming(wlen, 'periodic'))/wlen; winLim = [0, 45];
for i = 1:length(dr)
    load([dr(i).folder, '\', dr(i).name])
    select_data = relevant_data(:,SELECT_CHANNELS);
    % Filter data:
    for ch = 1:length(SELECT_CHANNELS)
        % Filter:
        f_data(:,ch) = customFilt(select_data(select,ch),Fs,[1.2, 41],3);
        %scale from uV to V:
        f_data(:,ch) = (f_data(:,ch)/1000000);
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
        [S, freq] = welch_psd(f_data(:,ch),Fs,hannWin);
        % STFT:
        [S1, f1, t1] = stft( f_data(:,ch), wlen, h, nfft, Fs ); S2 = 20*log10(abs(S1(f1<winLim(2) & f1>winLim(1),:))/wlen/K + 1e-6);
        if(plot_data)
            figure(ch)
            subplot(2,2,[1,2]); plot(freq,S), xlim(winLim);
            subplot(2,2,[3,4]); imagesc(t1,f1(f1<winLim(2) & f1>winLim(1)),S2),xlim([min(t1) max(t1)]),ylim(winLim);
            set(gca,'YDir','normal');xlabel('Time, s');ylabel('Frequency, Hz');colormap(jet);
            cb = colorbar;ylabel(cb, 'Power (db)')
        end
    end
    plot_data = input('Continue? \n');
    if(isempty(plot_data))
        plot_data = 1;
    end
    clf([1:5]);
end