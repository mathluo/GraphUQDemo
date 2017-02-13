% test GL and MBO on two moons data.  

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
Neig = 2000; 
V = V(:,1:Neig); 
E = E(1:Neig); 

%% Run Ginzburg Landau iteration
disp('Running MBO...'); 
tic;
% Do the GL Experiment
max_iter=50;
inner_iter = 5; 
dt = 2; 
eta = 1; 
u0 = sign(randn(N,1)); 
% u0 = sign(V(:,2)); 
disp(['dt: ', num2str(dt), ' eta: ', num2str(eta)]); 
[u_prediction, iter_stats] = MBO_binary(V, E, u0, eta, max_iter, inner_iter, dt, fid, ground_truth); 
time_ = toc; 

disp(['Running MBO took ', num2str(time_),  ' s']);

%% Display Results
close all; 

figure; 
plot(u_prediction);
title('predicted label values from MBO'); 







