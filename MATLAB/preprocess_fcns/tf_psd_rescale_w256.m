function [ Y ] = tf_psd_rescale_w256( X )
%tf_psd_rescale_w256 TF Preprocessing
% input should be X = (256, 2), or X = (512, 1);
% X = single(X); 
Fs = 250;
Y = single(zeros(2, 128));
if numel(X) == 512
    if size(X,1) == 512
        X = reshape(X, [256, 2]);
    elseif size(X,2) == 256
        X = X'; % if input is X = (2, 256), transpose
    end
end

Y(001:128) = rescale_minmax(tf_welch_psd(X(:,1), Fs, hannWin(256)));

Y(129:end) = rescale_minmax(tf_welch_psd(X(:,2), Fs, hannWin(256)));

% for ch = 1:2
%    Y(ch,:) = tf_welch_psd(X(:,ch), Fs, hannWin(256)); %
%    Y(ch,:) = rescale_minmax(Y(ch,:));
% end

end

