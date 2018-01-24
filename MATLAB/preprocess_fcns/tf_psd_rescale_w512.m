function [ Y ] = tf_psd_rescale_w512( X )
%tf_psd_rescale_w256 TF Preprocessing
% input should be X = (512, 2), or X = (1024, 1);
% Output is Y = (2, 256) float32
% X = single(X); 

% Y = single(zeros(2, 256));
Y = single(zeros(1, 256*2));
if numel(X) == 1024
    if size(X,1) == 1024
        X = reshape(X, [512, 2]);
    elseif size(X,2) == 512
        X = X'; % if input is X = (2, 256), transpose
    end
end

Y(001:256) = rescale_minmax(tf_welch_psd(X(:,1), 250, hannWin(512)));
Y(257:end) = rescale_minmax(tf_welch_psd(X(:,2), 250, hannWin(512)));

% for ch = 1:2
%    Y(256*(ch-1)+1:256*(ch-1)+256) = tf_welch_psd(X(:,ch), 250, hannWin(512)); %
%    Y(256*(ch-1)+1:256*(ch-1)+256) = rescale_minmax(Y(256*(ch-1)+1:256*(ch-1)+256));
% end

end

