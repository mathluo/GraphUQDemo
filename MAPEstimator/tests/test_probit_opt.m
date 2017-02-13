% test the probit optimization on two moons


sigma = 0.04;
fid_perc = 0.03;
tic;
disp('Generating Graph and Data...');
tau = 0.15; % fixed here
N = 1000;
data_params.n = N;
data_params.sigma = sqrt(sigma);
data_params.type = 'two_moons';
data = synthetic_data(data_params)';
ground_truth = zeros(N,1);
ground_truth(1:N/2) = 1;
ground_truth(N/2+1:N) = -1;
disp(['Number of points: ', num2str(data_params.n)]);
dist_mat_l2 = sqdist(data', data');
opt = {};
opt.graph = 'full';
opt.type = 's';
opt.tau = 2*tau^2;
LS = dense_laplacian(dist_mat_l2, opt);
dist_mat_l2 = sqdist(data', data');

% generate fidelity points
fidelity_percent = fid_perc; 
fid{1} = randi([1,N/2], ceil(N/2.0*fidelity_percent), 1);
fid{2} = randi([N/2+1,N], ceil(N/2.0*fidelity_percent), 1);
disp(['Number of points: ', num2str(N)]);
disp(['Percent of fidelity: ', num2str(fidelity_percent)]);
%% Compute Eigenvectors
[V,S]=eig(LS);
S = diag(S);

%% Do Probit PCN MCMC (This works fine now).
disp('Testing...');
tic;
beta=0.5; % proposal variance/step
gamma=0.2; g2=2.0*gamma*gamma; % obs noise std
max_iter=50; % number of mcmc steps
scale = probit_normalization_scale(S);
effective_eta = 1/(scale*gamma*gamma);
disp(['effective eta is: ', num2str(effective_eta)]);
E = S*scale;
E(1) = 1e6;
P = V * diag(E) * V';
u0 = 0.1*randn(N, 1);

% Newton Iteration
[m1,~] = probit_optimization_newton(P, u0 ,gamma,max_iter,fid);
time_ = toc;
disp(['Running Newton took ', num2str(time_),  ' s']);

% Forward Backward
dt = 1; 
max_iter = 1000; 
Neig = 1000; 
E = E(1:Neig); 
tic; 
[m2,~] = probit_optimization_eig(V,E, u0, dt,gamma,max_iter,fid);
time_ = toc;
disp(['Running Eig 1000 took ', num2str(time_),  ' s']);

% Forward Backward
Neig = 400; 
E = E(1:Neig); 
V = V(:, 1:Neig); 
tic; 
[m3,~] = probit_optimization_eig(V,E, u0, dt,gamma,max_iter,fid);
time_ = toc;
disp(['Running Eig 400 took ', num2str(time_),  ' s']);

% Forward Backward
Neig = 100; 
E = E(1:Neig); 
V = V(:, 1:Neig); 
tic; 
[m4,~] = probit_optimization_eig(V,E, u0, dt,gamma,max_iter,fid);
time_ = toc;
disp(['Running Eig 100 took ', num2str(time_),  ' s']);


figure; 
plot(m1, 'r+'); 
hold on; 
plot(m2, 'b+'); 
hold off; 

figure; 
plot(m3, 'r+'); 

figure; 
plot(m4, 'r+'); 







