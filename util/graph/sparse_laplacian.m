function L = sparse_laplacian(W,opt)

%--------------------------------------------------------------------------
% Author: Xiyang Luo <xylmath@gmail.com> , UCLA 
%
% This file is part of the diffuse-interface graph algorithm code. 
% There are currently no licenses. 
%
%--------------------------------------------------------------------------
% Description : Compute graph Laplacian from distance matrix D
%               Suitable only for small dense matrices. 
%
% Input: 
%       W : Sparse weight matrix
%       Opt : Options
%               type : 'u'(unnormalized), 's'(symmetric), 'rw'(random walk)
%--------------------------------------------------------------------------

opt = check_opt(opt); 
D = sum(W,1); 
m = size(W,1); 
if(strcmp(opt.type,'s'))
    D_inv_sqrt = 1./sqrt(D);
    D_inv_sqrt = spdiag(D_inv_sqrt,m,m); 
    L = speye(size(W,1)) - D_inv_sqrt*W*D_inv_sqrt; 
end
if(strcmp(opt.type,'rw'))
    D_inv = 1./D;
    D_inv = spdiag(D_inv,m,m); 
    L = speye(size(W,1)) - D_inv*W;     
end
if(strcmp(opt.type,'u'))
    D = spdiag(D); 
    L = D - W; 
end

end

function M = spdiag(D,m,n)
M = sparse(1:1:m,1:1:n,D); 
end

function opt = check_opt(opt)
    if(~isfield(opt, 'type')); 
        opt.type = 's'; 
    end
end
