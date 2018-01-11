function [ Y ] = tf_psd_rescale_w384( X )
%tf_psd_rescale_w256 TF Preprocessing
% input should be X = (384, 2), or X = (768, 1);
% Output is Y = (2, 192) float32
% X = single(X); 

Y = single(zeros(2, 192));
if numel(X) == 768
    if size(X,1) == 768
        X = reshape(X, [384, 2]);
    elseif size(X,2) == 384
        X = X'; % if input is X = (2, 256), transpose
    end
end

for ch = 1:2
   Y(ch,:) = tf_welch_psd(X(:,ch), 250, hannWin(384)); %
   Y(ch,:) = rescale_minmax(Y(ch,:));
end

end

