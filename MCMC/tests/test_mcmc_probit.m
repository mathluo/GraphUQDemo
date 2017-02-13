% script for testing MCMC algorithms on voting data. 

%% Generate the data and the graph
[ data, ground_truth, fid ] = voting_data('default', 16); 
dist_mat = sqdist(data', data'); 
opt = {}; 
opt.graph = 'rbf'; 
sigma = 0.45; 
%sigma = 0.8; 
%sigma = 1; 
opt.type = 's'; 
opt.tau = 2*sigma^2; 
LS = dense_laplacian(dist_mat, opt); 


% % visualize eigenvectors 
% [V,S]=eig(LS);
% figure; 
% for i = 1:5
%     for j = 1:5
%         p = (i-1)*5+j; 
%         subplot(5,5,p);
%         plot(V(:,p)); 
%     end
% end
% figure; 
% plot(diag(S)); 
        

%% Compute matrices
[V,S]=eig(LS);
S = diag(S); 
[P,G1,~] = normalize_dense(S,V); 


%% Test MCMC 
disp('Starting MCMC...'); 

% Do the MCMC Probit Experiment
beta=0.5; % proposal variance/step
gamma=0.001; g2=2.0*gamma*gamma; % obs noise std
max_iter=10^4; % number of mcmc steps
[m1,iter_stats1] = mcmc_probit_pcn(beta,gamma,max_iter,G1,P,V,fid,ground_truth); 


%% Plotting
movie_nodes_mcmc(iter_stats.y_mean_rec, iter_stats.u_rec,...
    iter_stats.pointwise_err_rec);  


