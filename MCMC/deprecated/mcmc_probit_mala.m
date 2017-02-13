function [ m,iter_stats ] = mcmc_probit_mala(dt,gamma,max_iter,sqtC,C,P,fid,ground_truth)
%--------------------------------------------------------------------------
%  Description : (Depreciated. Use Scale invariant version instead)
%       Perform MCMC(MALA) on Probit(only for binary labels)
%
%  Input : 
%       dt: stepsize for Langevine dynamics
%       gamma: scalar, noise of y
%       max_iter: maximum of iteration for MCMC
%       sqtC: sqrt covariance matrix, i.e. sqrt(C), i.e. L^{-1/2}  
%       C: covariance matrix, L^-1 
%       P: precision, i.e. the graph Laplacian L
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

N = size(sqtC,1);
g3=gamma*gamma; div=sqrt(2.0*pi*g3); % obs noise
mult = 1./(1+dt); 
sqdt = sqrt(dt); 

cc=zeros(N,1); % preallocate space
acc=zeros(max_iter,1); ct=zeros(max_iter,1); 
u_rec=zeros(N,max_iter); 
y_mean_rec=zeros(N,max_iter); 
pointwise_err_rec = zeros(N,max_iter); 

ctt=0.0; % initialize count number of acceptances
ratio=zeros(max_iter,1); % initialize energy and assignment
m=zeros(N,1);
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

% starting main iteration. 
for k=1:max_iter
    % compute grad\phi(u)
    du=zeros(N,1);    
    for i = 1:size(fid{1},1)
        du(fid{1}(i)) = grad_probit(u(fid{1}(i)),1, g3, div, gamma);
    end
    for i = 1:size(fid{1},1)
        du(fid{2}(i)) = grad_probit(u(fid{1}(i)),-1, g3, div, gamma);
    end    
    
    % compute stats for u
    pu = 1.0;  
    for i = 1:size(fid{1},1)
        pu = pu * likelihood_fn(u(fid{1}(i)),1);
    end
    for i = 1:size(fid{2},1)
        pu = pu * likelihood_fn(u(fid{2}(i)),-1);
    end
    P_m_u = P*u; 
    log_prob_u=-0.5*dot(u,P_m_u)+log(pu);
    
    %compute proposed move
    v=mult*(u-dt*C*du + sqdt*sqrt(2)*sqtC*z(:,k)); 
    %compute stats for v
    pv = 1.0;
    for i = 1:size(fid{1},1)
        pv = pv * likelihood_fn(v(fid{1}(i)),1);
    end
    for i = 1:size(fid{2},1)
        pv = pv * likelihood_fn(v(fid{2}(i)),-1);
    end
    P_m_v = P*v; 
    log_prob_v=-0.5*dot(v,P_m_v)+log(pv);    
    % compute acceptance probability
    wu = v - mult*(u+dt*C*du);
    log_q_vu = -.25*(1+dt)*(1+dt)/dt*dot(wu,P*wu); 
    dv=zeros(N,1);    
    for i = 1:size(fid{1},1)
        dv(fid{1}(i)) = grad_probit(v(fid{1}(i)),1, g3, div, gamma);
    end
    for i = 1:size(fid{1},1)
        dv(fid{2}(i)) = grad_probit(v(fid{1}(i)),-1, g3, div, gamma);
    end     
    wv = u - mult*(v+dt*C*dv); 
    log_q_uv = -.25*(1+dt)*(1+dt)/dt*dot(wv,P*wv); 
    
    % accept reject
    r=rand;
    p_acc = exp(log_prob_v + log_q_uv - log_prob_u - log_q_vu); 
    acc(k)=min(1,p_acc);
    ctt=((k-1)*ctt+acc(k))/k;
    ct(k)=ctt;
    if acc(k)>r
        u=v;
    end
    
    % compute iteration stats
    m=((k-1)*m+sign(u))/k;
    u_rec(:,k)=u;
    y_mean_rec(:,k)=m;
    diff1 = (sign(u_rec(:,k)) ~= ground_truth); 
    ratio(k) = 1-sum(diff1)/N; 
    cc=((k-1)*cc+diff1)/k;
    pointwise_err_rec(:,k)=cc;    
end

% record iteration stats
iter_stats.u_rec = u_rec; 
iter_stats.y_mean_rec = y_mean_rec; 
iter_stats.pointwise_err_rec = pointwise_err_rec; 
iter_stats.ratio = ratio; 
iter_stats.ct = ct; 
end

function res = pointwise_likelihood(x,y,gamma)
res = normcdf(x*y,0,gamma);  
end

function d = grad_probit(x,y,g3,div,gamma)
d=-y*exp(-x*x/(2.0*g3))/div;
d=d/normcdf(x,0,gamma);
end





