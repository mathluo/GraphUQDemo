% Demo on the Voting Records Dataset


%% Load data and Build Graph
[ data, ground_truth, fid ] = voting_data('default', 16);  
N = size(data, 1); 
dist_mat = sqdist(data', data');
opt = {}; opt.graph = 'rbf'; opt.type = 's';
sigma = 1.25;
opt.tau = 2*sigma^2;
LS = dense_laplacian(dist_mat, opt);

[V,S]=eig(LS);
S = diag(S);

figure; 
plot(S);  
acc_sp = sum(sign(V(:,2)) ~= ground_truth)/N; 
if(acc_sp < 0.5)
    acc_sp = 1 - acc_sp; 
    V(:, 2) = -V(:, 2); 
end

figure; 
plot(V(:, 2), 'b+');
title('Second Eigenvector'); 
disp(['Accuracy by Spectral Clustering : ', num2str(acc_sp)]); 

%% Test MCMC

% Probit
disp('Starting MCMC...');
tic;
beta=0.4; % proposal variance/step
gamma= 0.1; % obs noise std
max_iter=1 * 10^4; % number of mcmc steps
opt = {}; opt.isrec_u = false;
[m,iter_stats] = mcmc_probit_pcn_eig(beta,gamma, max_iter,V,S,fid, opt);
time_ = toc;
disp(['Running MCMC took ', num2str(time_),  ' s']);

acc_p = 1 - sum(sign(m) ~= ground_truth)/N; 
disp(['Accuracy by Probit : ', num2str(acc_p)]); 
plot(m, 'r+'); 
title('Probit'); 










