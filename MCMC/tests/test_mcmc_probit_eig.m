% script for testing MCMC algorithms on voting data. 

%% Generate the data and the graph
[ data, ground_truth, fid ] = voting_data('default', 16); 
dist_mat = sqdist(data', data'); 
opt = {}; 
opt.graph = 'rbf'; 
%sigma = 0.45; 
%sigma = 0.8; 
sigma = 1; 
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
Neig = 120; 
S = S(1:Neig); 
V = V(:,1:Neig); 
figure; 
plot(S); 


%% Test MCMC 

% Do the MCMC Probit Eig Experiment
disp('Starting MCMC...'); 
tic; 
beta=0.5; % proposal variance/step
gamma=0.1; % obs noise std
max_iter=10^4; % number of mcmc steps
scale = max(S); 
E = S/scale; 
[m2,iter_stats2] = mcmc_probit_pcn_eig_fill(beta,gamma,max_iter,V,E,fid); 
time_ = toc; 
disp(['Running MCMC took ', num2str(time_),  ' s']);

% plot results


