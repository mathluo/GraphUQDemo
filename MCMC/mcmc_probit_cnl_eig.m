function [ y_mean,iter_stats ] = mcmc_probit_cnl_eig(beta,gamma,max_iter,V,E,fid,varargin)
%--------------------------------------------------------------------------
%  Description : 
%       Perform MALA type algorithm(CNL) for Probit using eigenvectors.
%       (Two Problems: 
%           (1), explicit gradient on fidelity is too steep 
%               which restricts stepsize. Needs to go implicit on the Probit
%               likelihood term
%           (2), numerical instability of exact Probit gradient
%               is too much. 
%  Proposal : 
%       (1) Jp = .5<u, Lu> + Phi(u), where Phi = -loglike 
%       (2) (1 + .5beta L)v = (1 - .5beta L)u - 2beta GradPhi(u) +
%                   sqrt(8beta) L^-1/2 w. 
%       (2) rho(u, v) see helper function
%       (3) acc = exp(rho(u,v) - rho(v,u)); 
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
y_mean=zeros(N,1);
u_hat_mean=zeros(Neig,1);
if opt.isrec_u  
    u_rec=zeros(N,max_iter); 
    y_mean_rec=zeros(N,max_iter); 
end

% sample in batch a Gaussian Noise. 
z_hat_all=randn(Neig,max_iter);

% initialized u
u=zeros(N,1); 
for i = 1:size(fid{1},1)
    u(fid{1}(i)) = 1;
end
for i = 1:size(fid{2},1)
    u(fid{2}(i)) = -1;
end

% Precompute E^-1/2 and force samples to lie in the orthogonal complement
% of e^1 = (1, 1, ... 1)
E(1) = 1; 
inv_sqrt_E = 1./sqrt(E);
inv_E = 1./E; 
inv_E(1) = 0; 
inv_sqrt_E(1) = 0; 
E(1) = 0; 
coef1 = 1./(1 + .5*beta*E); 
coef1(1) = 0; 
coef2 = (1 - .5*beta*E).*coef1; 

% Main iteration. 
for k=1:max_iter
    % compute proposed move for v. 
    zk_hat = z_hat_all(:,k); 
    u_hat = V'*u; 
    grad_phi_u = grad_phi(u, fid, gamma); 
    grad_phi_u_hat = V'*grad_phi_u; 
    v_hat = coef2 .* u_hat - 2*beta*coef1.*grad_phi_u_hat + sqrt(8*beta)*inv_sqrt_E.*zk_hat; 
    v = V*v_hat; 
    % compute acceptance probability
    grad_phi_v = grad_phi(v, fid, gamma); 
    grad_phi_v_hat = V'*grad_phi_v; 
    rho_uv = rho_func(u, u_hat, v_hat, grad_phi_u_hat, inv_E, fid, gamma, beta); 
    rho_vu = rho_func(v, v_hat, u_hat, grad_phi_v_hat, inv_E, fid, gamma, beta); 
    acceptance_prob(k) = min(1, exp(rho_uv - rho_vu)); 
    % accep-reject step
    r=rand;
    if acceptance_prob(k)>r
        u=v;
    end    
    % updating and recording all variables 
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


function res = rho_func(u, u_hat, v_hat, grad_hat, inv_E, fid, gamma, beta)
    logpu = probit_data_log_likelihood(u, fid{1},1, gamma); 
    logpu = logpu + probit_data_log_likelihood(u, fid{2},-1, gamma); 
    res = -logpu + .5 * dot(v_hat - u_hat, grad_hat) +...
            beta/4 * dot(inv_E.*grad_hat, u_hat + v_hat) + ...
            beta/4 * dot(grad_hat', grad_hat);   
end

function res = grad_phi(u, fid, gamma)
    res = probit_data_log_likelihood_grad(u, fid{1},1, gamma); 
    res = res + probit_data_log_likelihood_grad(u, fid{2},-1, gamma); 
end



