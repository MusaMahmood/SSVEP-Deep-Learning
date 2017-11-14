%Large Collection:
clear;clc;close all;
% file_list_1 = dir('MAMEM_Large\EEG-SSVEP-Part1\*.mat'); % L x 1 struct
% file_list_2 = dir('MAMEM_Large\EEG-SSVEP-Part2\*.mat');
file_list_small = dir('MAMEM_Small_14ch\*.mat');
label_set_1 = [5 3 2 1 4 5 2 1 4 3];
label_set_2 = [4 2 3 5 1 2 5 4 2 3 1 5 4 3 2 4 1 2 5 3 4 1 3 1 3];

% load each file in "small"
for i = 1:length(file_list_small)
    filename = file_list_small(i).name;
    % Open:
    if(~file_list_small(i).isdir) 
        load([file_list_small(i).folder, '\', filename]);
        % reshape eeg to be in columns:
        if size(eeg,1) < size(eeg,2)
            eeg = eeg(:,:)'; %1:end-1
        end
        %use events to label 5 second fragments
%         if length(events) < 35
%             error('events should contain @ least 35 pts');
%         end
        
        filename_new = [file_list_small(i).folder, '\', filename(1:end-3) '.csv']
        %save as .csv
        
%         csvwrite(
    end
end