function [CSM,frequencies] = psd_custom(signals,fs,window)
if size(signals,2) > size(signals,1)
    signals = signals.';
end
% signals = signals(:);
window = window(:);
sensors = size(signals,2);
windowsize = length(window);
frequencies = (0:(windowsize/2-1))*fs/windowsize;
block_samples = windowsize; %must be even, best if 2^n
signal_samples = size(signals,1);
number_of_signals = size(signals,2);
back_shift = block_samples./2; %ORIGINAL;
number_of_blocks = floor((2*signal_samples)./block_samples) -1;
data_taper = window;
data_taper = repmat(data_taper,1,number_of_signals);
% Data segmentation into blocks of size block_samples:
S = zeros(block_samples/2,number_of_signals.^2); %ORIGINAL
% S = zeros(ceil(block_samples/2),number_of_signals.^2);
for a = 1:number_of_blocks
    % Retrieve current data block
    Data_Block = signals((a-1)*back_shift+1:block_samples +(a-1)*back_shift,:);
    Data_Block = Data_Block - repmat(mean(Data_Block),block_samples,1);
    Data_Block = Data_Block.*data_taper; %Taper it
    Data_Block = fft(Data_Block); %FFT it,
    % bilateral DFT
    % viii
    Data_Block = Data_Block(1:block_samples/2,:); %ORIGINAL
    % Data_Block = Data_Block(1:ceil(block_samples/2),:);
    %All spectral combinations:
    P = zeros(block_samples/2,number_of_signals.^2); %ORIGINAL
    % P = zeros(ceil(block_samples/2)/2,number_of_signals.^2);
    c = 1;
    for aa = 1:size(Data_Block,2)
        for b = aa:size(Data_Block,2)
            % THIS IS FOR WIND TUNNEL EESC-USP BEAMFORMING CODE
%             P(:,c) = real(Data_Block(:,b).*conj(Data_Block(:,aa))); 
            % P(:,c) = Data_Block(:,b).*conj(Data_Block(:,aa)); 
            % IS FOR FAN RIG BEAMFORMING CODE
            P(:,c) = real(Data_Block(:,aa).*conj(Data_Block(:,b)));
            % P(:,c) = Data_Block(:,aa).*conj(Data_Block(:,b)); % THIS IS THE ORIGINAL LINE
            c = c+1;
        end
    end
    % Sum the spectrums up ...
    S = S + P;
end
S = S*2/(sum(window.^2)*fs*number_of_blocks); % Average them out
Sf = zeros(sensors,size(S,1));
c=1;
% for a = 1:sensors
    for b = 1:sensors
        Sf(b,:) = S(:,c);
        c = c+1;
    end
% end
% clear S
CSM = Sf; 

for i = 1:size(CSM,1)
%     CSM(:,i) = CSM(:,i) + CSM(:,i)' - eye(sensors).*CSM(:,i);
    TEMP(i,:) = CSM(i,:) + CSM(i,:)' - eye(sensors).*CSM(:,i);
end


end