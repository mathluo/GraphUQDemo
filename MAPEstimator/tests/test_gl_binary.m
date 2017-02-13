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

%% Run Ginzburg Landau iteration
disp('Running Ginzburg-Landau...'); 
tic;
% Do the GL Experiment
max_iter=1000;
dt = .08; 
eps = 1; 
eta = 1.5; 
u0 = .07*randn(N,1); 
disp(['dt: ', num2str(dt), ' epsilon: ', num2str(eps), ' eta: ', num2str(eta)]); 
[u_prediction, iter_stats] = GL_binary(V, E, u0, eps, eta, max_iter, dt, fid, ground_truth); 
time_ = toc; 

disp(['Running Ginzburg-Landau took ', num2str(time_),  ' s']);

%% Display Results
close all; 

figure; 
plot(u_prediction);
title('predicted label values from Ginzburg-Landau'); 







