%% EEG Topographic Map generation
% S2 (SJ)
clear;clc;close all;
CH1_to_32 = [637.66888428  634.80163574  637.10656738  643.53027344  641.83624268 642.95153809  645.83203125  650.53723145  656.02703857  656.19348145 650.83746338  652.52758789  656.18249512  652.87207031  652.97735596 655.30615234  660.88421631  665.59320068  661.19403076  664.91491699 667.28009033  663.15386963  663.67828369  665.35968018  667.63427734 665.89941406  660.30566406  661.34802246  663.42205811  658.34777832 652.11468506  654.92034912];
CH33_to_64 = []; 
% for i = 1:32
%     CH1_to_32(i) = min(CH33_to_64);     
% end
DATA = [CH1_to_32, CH33_to_64];
figure; topoplot(rescale_minmax(DATA, -1, 1), 'Tsinghua_64-chs.loc','style','map','electrodes','labelpoint');
cb = colorbar;
% figure;topoplot([],'Chan64-setup.asc',[lo.hi],'style','map','electrodes','labelpoint'); %To plot channel locations only


