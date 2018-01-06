function [ Y ] = rescale_minmax( X, min_bound, max_bound )
if (nargin == 1)
    min_bound = 0;
    max_bound = 1;
end

minv = min(X(:));
maxv = max(X(:));

Y = min_bound + ( (X - minv) ./ (maxv - minv) ) .* (max_bound - min_bound);

end

