function [ y_mean,iter_stats ] = mcmc_probit_vanilla_eig(beta,gamma,max_iter,V,E,fid,varargin)
%--------------------------------------------------------------------------
%  Description : (works, but for a smaller beta than pCN). 
%       Vanilla sampling algorithm for Probit using eigenvector projection.
%  Proposal : 
%           (1) v = u + beta L^-1/2 w
%           (2) log(acc) = loglike(v) - loglike(u) + .5 * <u,Pu> -
%                           .5*<v,Pv>
%
%  Input : 
%       beta: scalar, proposal variance for MCMC 
%       gamma: scalar, noise of y
%       max_iter: maximum of iteration for MCMC
%       V: eigenvectors of L. (used to compute iteration stats only). 
%       E: normalized eigenvalues. 
%       fid : 2x1 cell array, indices of fidelity points for each class
%       varargin: opt: optional arguments. 
%           isrec_u: bool, whether to record u in the spatial domain. 
%                           Default True.  
%
%  Output : 
%       y_mean : mean of the prediction y for each label
%       iter_stats : stats for the MCMC iterations. Details below:
%                           vvv
%       y_mean_rec : record of the mean of the y-prediction. 
%       u_rec : record of the current sample of u. 
%       u_hat_rec : record of u_hat, i.e., coefficient on eigenvectors
%       mcmc_cum_prob : cummulative acceptance probability of the MCMC.
%       mcmc_100_prob : acceptance prob in 100 samples. 
% -------------------------------------------------------------------------

% default options
if(size(varargin) == 0)
    opt = {}; 
else 
    opt = varargin{1}; 
end
opt = check_opt(opt); 
% initialize variables
N = size(V,1);
Neig = size(V,2); 
acceptance_prob=zeros(max_iter,1); 
mcmc_cum_prob=zeros(max_iter,1); 
u_hat_mean_rec=zeros(Neig,max_iter); 
u_hat_rec=zeros(Neig,max_iter); 
energy=zeros(max_iter,1); 
y_mean=zeros(N,1);
u_hat_mean=zeros(Neig,1);
if opt.isrec_u  
    u_rec=zeros(N,max_iter); 
    y_mean_rec=zeros(N,max_iter); 
end

% sample in batch a Gaussian Noise. 
z_hat_all=randn(Neig,max_iter);

% initialized u by 1, -1 assignment
u=zeros(N,1); 
for i = 1:size(fid{1},1)
    u(fid{1}(i)) = 1;
end
for i = 1:size(fid{2},1)
    u(fid{2}(i)) = -1;
end

% % initialized u by rand
% u=randn(N,1); 

% Precompute E^-1/2 and force samples to lie in the orthogonal complement
% of e^1 = (1, 1, ... 1)
E(1) = 0; 
inv_sqrt_E = 1./sqrt(E); 
inv_sqrt_E(1) = 0; 

% Main iteration. 
for k=1:max_iter
    logpu = 0; 
    logpv = 0; 
    % compute proposed move for v. 
    zk_hat = z_hat_all(:,k); 
    temp1 = zk_hat.*inv_sqrt_E; 
    v = u + beta*V*temp1; 
    % compute acceptance probability
    u_hat = V' * u; 
    eu = .5*dot(u_hat,E.*u_hat); 
    v_hat = u_hat + beta*temp1; 
    ev = .5*dot(v_hat,E.*v_hat); 
    logpu = logpu + probit_data_log_likelihood(u, fid{1},1, gamma); 
    logpu = logpu + probit_data_log_likelihood(u, fid{2},-1, gamma); 
    logpv = logpv + probit_data_log_likelihood(v, fid{1},1, gamma); 
    logpv = logpv + probit_data_log_likelihood(v, fid{2},-1, gamma); 
    log_acc = logpv - logpu + eu - ev; 
    acceptance_prob(k) = min(1, exp(log_acc)); 
    % accep-reject step
    r=rand;
    if acceptance_prob(k)>r
        u=v;
    end    
    % updating and recording all variables 
    energy(k) = .5*dot(u_hat,E.*u_hat)-logpu; 
    if k == 1
        mcmc_cum_prob(k) = acceptance_prob(k); 
    else
        mcmc_cum_prob(k)=((k-1)*mcmc_cum_prob(k-1)+acceptance_prob(k))/k;
    end
    y_mean=((k-1)*y_mean+sign(u))/k;
    u_rec(:,k)=u;
    y_mean_rec(:,k)=y_mean;
    u_hat_rec(:,k)=u_hat;
    u_hat_mean=((k-1)*u_hat_mean+u_hat_rec(:,k))/k;
    u_hat_mean_rec(:,k)=u_hat_mean; 
end

% record iteration results to iter_stats. 
if opt.isrec_u
    iter_stats.u_rec = u_rec; 
    iter_stats.y_mean_rec = y_mean_rec; 
end
iter_stats.u_hat_rec = u_hat_rec; 
iter_stats.u_hat_mean_rec = u_hat_mean_rec; 
iter_stats.mcmc_cum_prob = mcmc_cum_prob; 
iter_stats.mcmc_100_prob = zeros(ceil(max_iter / 100), 1); 
for i = 1:size(iter_stats.mcmc_100_prob, 1)
   iter_stats.mcmc_100_prob(i) = mean(acceptance_prob((i-1)*100 + 1 : min(i * 100, max_iter)));  
end
end

function opt = check_opt(opt)
if(~isfield(opt,'isrec_u'))
    opt.isrec_u = true; 
end
end



