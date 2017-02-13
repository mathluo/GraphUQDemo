% construct graphs for for MNIST using method in Hu et al paper
% pca the digits to 50 dimensions
% pick 10 nearest neighbors as graphs
% compute using Perona Scaling.

function [V, E, ground_truth] = makemnistgraph(digits, num_points, Neig, Ltype)

num_components = 50;
num_nbs = 10;
[lb, im] = mnist_data(digits);

% subsample a smaller portion, 40 % totally 4000 points
N1 = sum(lb == digits(1));
N2 = sum(lb == digits(2));
n1 = num_points(1); 
n2 = num_points(2); 
ind1 = randperm(N1, n1);
ind2 = randperm(N2, n2) + N1;
ind = [ind1, ind2];
lb = lb(ind);

% do PCA onto 50 dimensions first.
% sc is the projection of the digits onto the axis. (N x k) matrix
[~, sc] = pca(im, 'NumComponents', num_components);
sc = sc(ind, :);
im = im(ind, :); 

% do 10 nearest neighbors, and mean dist between 10 nearest neighbors
num_points = size(sc, 1);
dist = sqdist(sc', sc');
[dist, index] = sort(dist, 2, 'ascend');
d_sp = dist(:, 2:num_nbs+1);
j_sp = index(:, 2:num_nbs+1);
clear dist index;

% compute the weights via the scaling by mean of closest 10 dist
dsum_sp = sum(d_sp, 2);
dmean_sp = dsum_sp / num_nbs;
w_sp = bsxfun(@rdivide, d_sp, dmean_sp);
w_sp = exp(-(w_sp .* w_sp)/ 3);


if Ltype == 'u'
    % compute and store sparse matrix L(unnormalized)
    i_sp = reshape((1:num_points)' * ones(1, num_nbs), 1, num_points * num_nbs , 1);
    j_sp = reshape(j_sp, numel(j_sp), 1);
    W = sparse(i_sp, j_sp, w_sp);
    W = .5 * (W + W'); % symmetrize
    wsum_sp = sum(W, 2);
    wsum_sp(wsum_sp < 1e-6) = 1e-6;
    isum_sp = 1:num_points;
    D = sparse(isum_sp, isum_sp, wsum_sp);
    Dsqrtinv =  sparse(isum_sp, isum_sp, 1./sqrt(wsum_sp));
    L = D - W;
end

if Ltype == 's'
    % compute and store sparse matrix L(symmetric)
    i_sp = reshape((1:num_points)' * ones(1, num_nbs), 1, num_points * num_nbs , 1);
    j_sp = reshape(j_sp, numel(j_sp), 1);
    W = sparse(i_sp, j_sp, w_sp);
    W = .5 * (W + W'); % symmetrize
    wsum_sp = sum(W, 2);
    wsum_sp(wsum_sp < 1e-6) = 1e-6;
    isum_sp = 1:num_points;
    D = sparse(isum_sp, isum_sp, wsum_sp);
    Dsqrtinv =  sparse(isum_sp, isum_sp, 1./sqrt(wsum_sp));
    L = speye(num_points) - Dsqrtinv * W * Dsqrtinv;
end

% compute the eigenvectors
[V, E] = eigs(L, Neig, 'SM');
E = diag(E);

ground_truth = ones(size(lb));
ground_truth(lb == digits(2)) = -1;


end




