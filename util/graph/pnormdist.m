function [D] = pnormdist(X, Y, p)
%--------------------------------------------------------------------------
% Author: Xiyang Luo <xylmath@gmail.com> , UCLA 
%
% This file is part of the diffuse-interface graph algorithm code. 
% There are currently no licenses. 
%
%--------------------------------------------------------------------------
% Description: Computes pairwise p-norm distance. p is allowed to be Inf
%
% Usage: 
%       X = (x1,x2,\dots, xm) m col vectors
%       Y = (y1,y2,\dots, yn) n col vectors
%       D_ij = ||xi-yj||_p (m x n) matrix
%--------------------------------------------------------------------------

m = size(X,2);
n = size(Y,2); 
D = zeros(m,n); 
if p == 2
   disp('Warning: Use sqdist for computing square norms');  
end

for i = 1:m
    for j = 1:n
        v = X(:, i) - Y(:, j); 
        D(i,j) = norm(v, p); 
    end
end    
end

