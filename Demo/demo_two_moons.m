% Demo on the Two Moons Dataset


%% Load data and Build Graph
tic;
disp('Generating Graph and Data...');
% generate data
sigma = 0.04;
N = 1000;
Neig = 100;
data_params.n = N;
data_params.sigma = sqrt(sigma);
data_params.type = 'two_moons';
data = zeros(N, 100);
data(:, 1:2) = synthetic_data(data_params)';
data(:, 3:100) = sigma * randn(N, 98);
ground_truth = zeros(N,1);
ground_truth(1:N/2) = 1;
ground_truth(N/2+1:N) = -1;
disp(['Number of points: ', num2str(data_params.n)]);
% build graph
dist_mat_l2 = sqdist(data', data');
opt = {};
opt.graph = 'full';
opt.type = 's';
tau = 0.3;
opt.tau = 2*tau^2;
LS = dense_laplacian(dist_mat_l2, opt);
% compute eigenvector
[V,S]=eig(LS);
% Note if I don't truncate here the accuracy decreases.
% Most of the eigenvalues are noise after some threshold. 
V = V(:, 1:Neig);  
S = diag(S);
S = S(1:Neig); 
% plot spectrum
figure;
plot(S);
acc_sp = sum(sign(V(:,2)) ~= ground_truth)/N;
if(acc_sp < 0.5)
    acc_sp = 1 - acc_sp;
    V(:, 2) = -V(:, 2);
end
% plot accuracy
figure;
plot(V(:, 2), 'b+');
title('Second Eigenvector');
disp(['Accuracy by Spectral Clustering : ', num2str(acc_sp)]);

%% Test MCMC
% generate fidelity points
fidelity_percent = 0.03;
fid{1} = randi([1,N/2], ceil(N/2.0*fidelity_percent), 1);
fid{2} = randi([N/2+1,N], ceil(N/2.0*fidelity_percent), 1);
disp(['Number of points: ', num2str(N)]);
disp(['Percent of fidelity: ', num2str(fidelity_percent)]);
% Probit
disp('Starting MCMC...');
tic;
beta=0.2; % proposal variance/step
gamma= 0.15; % obs noise std
max_iter=1 * 10^4; % number of mcmc steps
opt = {}; opt.isrec_u = false;
[m,iter_stats] = mcmc_probit_pcn_eig(beta,gamma, max_iter,V,S,fid, opt);
time_ = toc;
disp(['Running MCMC took ', num2str(time_),  ' s']);

acc_p = 1 - sum(sign(m) ~= ground_truth)/N;
disp(['Accuracy by Probit : ', num2str(acc_p)]);
plot(m, 'r+');
title('Probit');










