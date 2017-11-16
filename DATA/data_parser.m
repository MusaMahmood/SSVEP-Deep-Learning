%Large Collection:
clear;clc;close all;
output_dir = 'output_csv\';
extension = '.csv';

for subject=1:11
    for session = 1:5
        sess = eegtoolkit.util.Session;
        if(sess.sessions{1,subject,session}(1:4)~='NULL')
            sess.loadSubjectSession(1,subject,session);
            %get filename:
            full_output_dir = [output_dir, sess.sessions{1,subject,session}(1:end-1), '\'];
            mkdir(full_output_dir);
            filename = [full_output_dir, sess.sessions{1,subject,session}];
            disp(length(sess.trials))
            for trial = 1:length(sess.trials)
                %fix label:
                sess.trials{trial}.label = double(getLabel(sess.trials{trial}.label));
                relevant_data = sess.trials{trial}.signal';
                relevant_data(:,end) = sess.trials{trial}.label;
                trial_num = ['_t',num2str(trial,'%03d')];
                % save to mat output folder:
                filename_out = [filename, trial_num];
                save([filename_out, '.mat'], 'relevant_data');
                disp(filename_out)
            end
        else
            
        end
    end
    %Destroy after each session?
    clear sess
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

