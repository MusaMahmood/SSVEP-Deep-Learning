%Large Collection:
clear;clc;close all;
output_dir = 'output_csv\';
extension = '.csv';
sess = eegtoolkit.util.Session;
%{
subject = 1;
for session = 1:3
    sess.loadSubjectSession(1,subject,session);
    %get filename:
    output_dir = ['output_csv\', sess.sessions{1,subject,session}(1:end-1), '\'];
    mkdir(output_dir);
    filename = [output_dir, sess.sessions{1,subject,session}];
    TRIALS = cell(sess.trials);
    for trial = 1:length(TRIALS)
        %fix label:
        TRIALS{trial}.label = double(getLabel(TRIALS{trial}.label));
        relevant_data = TRIALS{trial}.signal';
        relevant_data(:,end) = TRIALS{trial}.label;
        trial_num = ['_t',num2str(trial,'%03d')];
        % save to CSV output folder:
        filename_out = [filename, trial_num];
%         dlmwrite([filename_out,extension], relevant_data, 'precision', '%5.16f');
        save([filename_out, '.mat'], 'relevant_data');
    end
end
%}
for subject=4:11
    for session = 1:5
        if(sess.sessions{1,subject,session}(1:4)~='NULL')
            sess.loadSubjectSession(1,subject,session);
            %get filename:
            output_dir = ['output_csv\', sess.sessions{1,subject,session}(1:end-1), '\'];
            mkdir(output_dir);
            filename = [output_dir, sess.sessions{1,subject,session}];
            TRIALS = cell(sess.trials);
            for trial = 1:length(TRIALS)
                %fix label:
                TRIALS{trial}.label = double(getLabel(TRIALS{trial}.label));
                relevant_data = TRIALS{trial}.signal';
                relevant_data(:,end) = TRIALS{trial}.label;
                trial_num = ['_t',num2str(trial,'%03d')];
                % save to mat output folder:
                filename_out = [filename, trial_num];
                save([filename_out, '.mat'], 'relevant_data');
                disp(filename_out)
            end
        else
            
        end
    end
end

function fixedLabel = getLabel(label)
    if(label>0 && label<6.9)
        fixedLabel = 1;
    elseif (label < 7.7)
        fixedLabel = 2;
    elseif (label < 8.925)
        fixedLabel = 3;
    elseif (label < 9.95)
        fixedLabel = 4;
    elseif (label < 12.66)
        fixedLabel = 5;
    end
end

% %{



%{ 
for i = 1:length(file_list_small)
    filename = file_list_small(i).name;
    % Open:
    if(~file_list_small(i).isdir) 
        result = load([file_list_small(i).folder, '\', filename]);
        % reshape eeg to be in columns:
        if size(eeg,1) < size(eeg,2)
            eeg = eeg(:,:)'; %1:end-1
        end

        
        filename_new = [file_list_small(i).folder, '\', filename(1:end-3) '.csv']
        %save as .csv
        
%         csvwrite(
    end
end

%}
%}

