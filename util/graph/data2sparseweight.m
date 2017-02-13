function W = data2sparseweight(data,opt)
%--------------------------------------------------------------------------
% Author: Xiyang Luo <xylmath@gmail.com> , UCLA 
%
% This file is part of the diffuse-interface graph algorithm code. 
% There are currently no licenses. 
%
%--------------------------------------------------------------------------
% Description : Compute sparse weight matrix for graph from raw data
%               Uses Kd-tree from vlfeat package for graph sparsification 
%
% Input: 
%       data : data matrix. (n_samples, n_features)
%       Opt : Options
%               type : 'u'(unnormalized), 's'(symmetric), 'rw'(random walk)
%               tau : ker
% Output: 
%       W : weight matrix
%--------------------------------------------------------------------------

kdtree = vl_kdtreebuild(data'); %randomize trees
k = opt.num_neighbors; 
[indices, sq_dist] = vl_kdtreequery(kdtree, data', data', ...
    'NumNeighbors', k, 'MaxComparisons', 256);
tau = opt.tau; 
r = size(data, 1);
I = repmat(1:r, k, 1);
I = I(:);             % row indices for the sparse matrix
J = double(indices(:)); % column indices for the sparse matrix
A=exp(-sq_dist/tau);
W = sparse(I, J, A);
W = .5*(W+W'); 
end

