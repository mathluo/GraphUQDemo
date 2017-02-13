function L = dense_laplacian(dist_mat,opt)

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
%       dist_mat : Pairwise distance matrix 
%       Opt : Options
%               type : 'u'(unnormalized), 's'(symmetric), 'rw'(random walk)
%               tau : kernel width
%               k : number of neighbors
%               graph : 'full', 'knearest', 'z-p'
%--------------------------------------------------------------------------


opt = check_opt(opt); 

if(strcmp(opt.graph, 'z-p'))
    W = dist_mat; 
    K = opt.k;
    inv_d = zeros(size(W,1),1);
    for i = 1:size(W,1)
        [~,ind] = sort(W(:,i),1,'ascend');
        inv_d(i) = 1/sqrt(W(ind(K),i));
    end
    W =bsxfun(@times, bsxfun(@times,inv_d,W),inv_d'); 
    W = exp(-W);
else
    W = exp(-dist_mat/opt.tau);
    if(strcmp(opt.graph, 'knearest'))
        K = opt.k;
        for i = 1:size(W,1)
            [~,ind] = sort(W(:,i),1,'descend');
            W(ind(K+1:size(W,1)),i) = 0;
        end
        W = .5*(W+W'); %symmetrize
    end
    
end
    
D = sum(W,1);     

if(strcmp(opt.type,'s'))
    D_inv_sqrt = 1./sqrt(D);
    D_inv_sqrt = diag(D_inv_sqrt); 
    L = eye(size(W,1)) - D_inv_sqrt*W*D_inv_sqrt; 
    L = .5*(L+L'); 
end
if(strcmp(opt.type,'rw'))
    D_inv = 1./D;
    D_inv = diag(D_inv); 
    L = eye(size(W,1)) - D_inv*W;     
end
if(strcmp(opt.type,'u'))
    D = diag(D); 
    L = D - W; 
    L = .5*(L+L'); 
end

end

function opt = check_opt(opt)
    if(~isfield(opt, 'type')); 
        opt.type = 's'; 
    end
    if(~isfield(opt, 'tau')); 
        opt.tau = 1; 
    end
end
