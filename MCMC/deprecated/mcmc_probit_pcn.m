function [ m,iter_stats ] = mcmc_probit_pcn(beta,gamma,max_iter,G,P,V,fid,ground_truth)
%--------------------------------------------------------------------------
%  Description : 
%       Perform MCMC(PCN) on Probit
%       (Deprecated!!!!!!)
%
%  Input : 
%       beta: scalar, proposal variance for MCMC 
%       gamma: scalar, noise of y
%       max_iter: maximum of iteration for MCMC
%       G: sqrt covariance matrix, i.e. sqrt(C), i.e. L^{-1/2} 
%       V: eigenvectors of L. (used to compute iteration stats only). 
%       P: precision, i.e. the graph Laplacian L.(used to compute energy)
%       fid : 2x1 cell array, indices of fidelity points for each class
%       ground_truth: Nx1 array, {-1,1} labelled vector for ground truth. 
%
%  Output : 
%       m : mean of the prediction y for each label
%       iter_stats : stats for the MCMC iterations. Details below:
%                           vvv
%       y_mean_rec : record of the mean of the y-prediction. 
%       u_rec : record of the current sample of u. 
%       pointwise_err_rec : record of mean prediction error from sample.
%       u_hat_rec : record of u_hat, i.e., coefficient on eigenvectors
%       ratio : overall classification rate per timestep
%       ct : acceptance probability of the MCMC. 
% -------------------------------------------------------------------------

ct=0.0; % count number of acceptances

N = size(G,1);

v=zeros(N,1); C=zeros(N,N); cc=zeros(N,1); % preallocate space
acc=zeros(max_iter,1); ct=zeros(max_iter,1); dt=zeros(N,max_iter); % preallocate space
u_rec=zeros(N,max_iter); diff=zeros(N,1); diffn=zeros(N,1); % preallocate space
y_mean_rec=zeros(N,max_iter); u_hat_mean_rec=zeros(N,max_iter); 
u_hat_rec=zeros(N,max_iter); et=zeros(N,max_iter); % preallocate space
pointwise_err_rec = zeros(N,max_iter); 

ctt=0.0; % initialize count number of acceptances
energy=zeros(max_iter,1); ratio=zeros(max_iter,1); % initialize energy and assignment
m=zeros(N,1);
u_hat_mean=zeros(N,1);
dn=zeros(N,1);
pv=1.0; pu=1.0;
z=randn(N,max_iter);


u=zeros(N,1); 
for i = 1:size(fid{1},1)
    u(fid{1}(i)) = 1;
end
for i = 1:size(fid{2},1)
    u(fid{2}(i)) = -1;
end
% set the likelihood function
likelihood_fn = @(x,y) pointwise_likelihood(x,y,gamma);

% various initializations of mcmc
%u(1:1:267)=1; u(268:1:end)=-1;
%u=sign(randn(N,1));
%u(16)=1; u(164)=1; u(271)=1; u(378)=-1; u(400)=-1;
for k=1:max_iter
    pu = 1.0; 
    pv = 1.0; 
    % 
    %%% pu data likelihood at current
    %
    for i = 1:size(fid{1},1)
        pu = pu * likelihood_fn(u(fid{1}(i)),1);
    end
    for i = 1:size(fid{2},1)
        pu = pu * likelihood_fn(u(fid{2}(i)),-1);
    end
    energy(k)=0.5*dot(u,P*u)-log(pu);
    %
    %%% v proposed move
    %
    v=sqrt(1-beta^2)*u+beta*G*z(:,k);
    %
    %%% pv data likelihood at proposed
    %
    for i = 1:size(fid{1},1)
        pv = pv * likelihood_fn(v(fid{1}(i)),1);
    end
    for i = 1:size(fid{2},1)
        pv = pv * likelihood_fn(v(fid{2}(i)),-1);
    end
    r=rand;
    %
    %%% acceptance probability and running mean of same
    %
    acc(k)=min(1,pv/pu);
    ctt=((k-1)*ctt+acc(k))/k;
    ct(k)=ctt;
    %
    %%% accept-reject step
    %
    if acc(k)>r
        u=v;
    end
    %
    %%% sign(u) is classification variable
    %%% m running mean of sign(u)
    %%% nt is transform of u into eigenbasis
    %%% nm is running mean of nt
    %%% C is running covariance of m and dt sqrt of diagonal entries
    %%% dn is running diagonal std of nt
    %
    m=((k-1)*m+sign(u))/k;
    u_rec(:,k)=u;
    y_mean_rec(:,k)=m;
    u_hat_rec(:,k)=V'*u;
    u_hat_mean=((k-1)*u_hat_mean+u_hat_rec(:,k))/k;
    u_hat_mean_rec(:,k)=u_hat_mean;
    diff=sign(u)-m;
    C=((k-1)*C+diff*diff')/k;
    dt(:,k)=sqrt(diag(C));
    diffn=u_hat_rec(:,k)-u_hat_mean;
    dn=((k-1)*dn+diffn.*diffn)/k;
    et(:,k)=sqrt(dn);
%     %
%     %%% ratio is proportion of correctly assigned vertices
%     %%% diff1 is assignation of vertex to 1 (if u=mt>0) or 0 (otherwise)
%     %%% cc is running mean of diff1 giving probability of assignment to 1
%     %
    diff1 = (sign(u_rec(:,k)) ~= ground_truth); 
    ratio(k) = 1-sum(diff1)/N; 
    cc=((k-1)*cc+diff1)/k;
    pointwise_err_rec(:,k)=cc;    
end

iter_stats.u_rec = u_rec; 
iter_stats.y_mean_rec = y_mean_rec; 
iter_stats.u_hat_rec = u_hat_rec; 
iter_stats.u_hat_mean_rec = u_hat_mean_rec; 
iter_stats.pointwise_err_rec = pointwise_err_rec; 
iter_stats.ratio = ratio; 
iter_stats.ct = ct; 
end









