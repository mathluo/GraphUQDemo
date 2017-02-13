function [ m,iter_stats ] = mcmc_threshold_pcn_eig(beta,max_iter,p,q,V,E,fid,ground_truth)
%--------------------------------------------------------------------------
%  Description : 
%       Perform MCMC on Threshold using eigenvectors.
%       (only for binary labels)
%
%  Input : 
%       beta: scalar, proposal variance for MCMC 
%       max_iter: maximum of iteration for MCMC
%       p: probability of missclassifying class 1, threshold params
%       q: probability of missclassifying class 2, threshold params
%       V: eigenvectors of L. (used to compute iteration stats only). 
%       E: normalized eigenvalues. 
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


N = size(V,1);
Neig = size(V,2); 


% initialize variables
v=zeros(N,1); cc=zeros(N,1); % preallocate space
acc=zeros(max_iter,1); ct=zeros(max_iter,1);  % preallocate space
u_rec=zeros(N,max_iter); 
y_mean_rec=zeros(N,max_iter); u_hat_mean_rec=zeros(Neig,max_iter); 
u_hat_rec=zeros(Neig,max_iter); 
pointwise_err_rec = zeros(N,max_iter); 
ctt=0.0; % initialize count number of acceptances
energy=zeros(max_iter,1); ratio=zeros(max_iter,1); 
m=zeros(N,1);
u_hat_mean=zeros(Neig,1);
z_hat_all=randn(Neig,max_iter);

u=zeros(N,1); %zero initialization for u
for i = 1:size(fid{1},1)
    u(fid{1}(i)) = 1;
end
for i = 1:size(fid{2},1)
    u(fid{2}(i)) = -1;
end


% set the likelihood function
likelihood_fn = @(x,y) pointwise_likelihood(x,y,p,q);
E(1) = 1e6; 
inv_sqrt_E = 1./sqrt(E); 
inv_sqrt_E(1) = 0; 
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
    
    %energy(k)=0.5*dot(u,P*u)-log(pu);
    u_hat = V'*u; 
    energy(k) = .5*dot(u_hat,E.*u_hat)-log(pu); 
    %
    %%% v proposed move
    %
    zk_hat = z_hat_all(:,k); 
    temp1 = zk_hat.*inv_sqrt_E; 
    v = sqrt(1-beta^2)*u + beta*V*temp1; 
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
    
    % calculating ground truth
%     %
%     %%% ratio is proportion of correctly assigned vertices
%     %%% diff1 is assignation of vertex to 1 (if u=mt>0) or 0 (otherwise)
%     %%% cc is running mean of diff1 giving probability of assignment to 1
%     %
    diff1 = (sign(u_rec(:,k)) ~= ground_truth); 
    ratio(k) = 1-sum(diff1)/N; 
    cc=((k-1)*cc+diff1)/k;
    pointwise_err_rec(:,k)=cc;
%     ratio(k)=sum(sign(u_rec(1:267,k)))-sum(sign(u_rec(268:end,k))); ratio(k)=(ratio(k)+435)/870;
%     diff1=[sign(u_rec(1:267,k))>0;sign(u_rec(268:N,k))>0];
%     cc=((k-1)*cc+diff1)/k;
%     dt1(:,k)=cc;
end
iter_stats.u_rec = u_rec; 
iter_stats.y_mean_rec = y_mean_rec; 
iter_stats.u_hat_rec = u_hat_rec; 
iter_stats.u_hat_mean_rec = u_hat_mean_rec; 
iter_stats.pointwise_err_rec = pointwise_err_rec; 
iter_stats.ratio = ratio; 
iter_stats.ct = ct; 
end

function res = pointwise_likelihood(x,y,p,q)
res = ff(x,y,p,q); % ff being the probability threshold. 
end

