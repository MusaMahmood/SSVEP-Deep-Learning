function [ Y ] = rescale_minmax( X, min_bound, max_bound )
% reduce dims:
% x_size = size(X);
% if length(x_size) == 3
%     zero_axes = x_size == 1;
%     final_dim = x_size(~zero_axes);
%     X = reshape(X, final_dim);
% end

if (nargin == 1)
    min_bound = 0;
    max_bound = 1;
end

minv = min(X(:));
maxv = max(X(:));

Y = min_bound + ( (X - minv) ./ (maxv - minv) ) .* (max_bound - min_bound);

end

