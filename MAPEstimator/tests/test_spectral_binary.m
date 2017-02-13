% test spectral clustering binary version on two moons data. 

%% Generate two moons data
% generate data points
disp('Generating Graph and Data...'); 
tic; 
data_params.n = 2000; 
data_params.type = 'two_moons'; 
data = synthetic_data(data_params)'; 
N = size(data,1); 
ground_truth = zeros(N,1); 
ground_truth(1:N/2) = 1; 
ground_truth(N/2+1:N) = -1; 
fidelity_percent = 0.03;

% generate fidelity points
fid{1} = randi([1,N/2], ceil(data_params.n/2.0*fidelity_percent), 1); 
fid{2} = randi([N/2+1,N], ceil(data_params.n/2.0*fidelity_percent), 1); 
dist_mat = sqdist(data', data'); 
disp(['Number of points: ', num2str(data_params.n)]); 
disp(['Percent of fidelity: ', num2str(fidelity_percent)]);

% generate graph Laplacian
opt = {}; 
opt.graph = 'rbf'; 
sigma = .3; 
opt.type = 's'; 
opt.tau = 2*sigma^2; 
LS = dense_laplacian(dist_mat, opt); 

%% Compute matrices
[V,S]=eig(LS);
S = diag(S); 
time_ = toc; 
disp(['Generation of Graph took ', num2str(time_),  ' s']);
E = S; 
Neig = 4; 
V = V(:, 1:Neig); 

%% Run Spectral clustering

[u_out1, stats1] = spectral_binary(V, ground_truth, 'threshold'); 
[u_out2, stats2] = spectral_binary(V, ground_truth, 'kmeans'); 
V = V(:, 1:3); 
[u_out3, stats3] = spectral_binary(V, ground_truth, 'kmeans'); 
V = V(:, 1:2); 
[u_out4, stats4] = spectral_binary(V, ground_truth, 'kmeans'); 

figure; 
plot(u_out1); 
title(['Classification result threshold, err = ', num2str(stats1.error)]) ; 


figure; 
plot(u_out4); 
title(['Classification result kmeans dim = 2, err = ', num2str(stats4.error)]) ; 

figure; 
plot(u_out3); 
title(['Classification result kmeans dim = 3, err = ', num2str(stats3.error)]) ; 

figure; 
plot(u_out2); 
title(['Classification result kmeans dim = 4, err = ', num2str(stats2.error)]) ; 





