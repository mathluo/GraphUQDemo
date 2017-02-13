% Demo on the MNIST datast

digits = [4, 9];
num_points = [2000, 2000];
LType = 's';
Neig = 300;
disp(['Building MNIST Graph']);
tic;
[V, E, ground_truth] = makemnistgraph(digits, num_points, Neig, LType);
disp(['Computing MNIST Graph took ', num2str(toc), 's']);
E = E./max(E);
Neig = size(E, 1);
N = numel(ground_truth);
% plot spectrum
figure;
plot(E);
acc_sp = sum(sign(V(:,3)) ~= ground_truth)/N;
if(acc_sp < 0.5)
    acc_sp = 1 - acc_sp;
    V(:, 3) = -V(:, 3);
end
disp(['Accuracy by Spectral Clustering : ', num2str(acc_sp)]);
% plot to visualize
figure;
plot(V(ground_truth > 0, 2), V(ground_truth > 0, 3), 'r.');
hold on;
plot(V(ground_truth < 0, 2), V(ground_truth < 0, 3), 'b.');
hold off;
title('2nd and 3rd eigenvector');

%% Test MCMC
% generate fidelity points
fidelity_percent = 0.04;
N1 = sum(ground_truth == 1);
N2 = sum(ground_truth == -1);
fid{1} = randi([1,N1], ceil(N1*fidelity_percent), 1);
fid{2} = randi([N1 + 1, N1 + N2], ceil(N2*fidelity_percent), 1);
disp(['Number of points: ', num2str(N)]);
disp(['Percent of fidelity: ', num2str(fidelity_percent)]);
% Probit
disp('Starting MCMC...');
tic;
beta=0.2; % proposal variance/step
gamma= 0.15; % obs noise std
max_iter= 8000; % number of mcmc steps
opt = {}; opt.isrec_u = false;
[m,iter_stats] = mcmc_probit_pcn_eig(beta,gamma, max_iter,V,E,fid, opt);
time_ = toc;
disp(['Running MCMC took ', num2str(time_),  ' s']);

acc_p = 1 - sum(sign(m) ~= ground_truth)/N;
disp(['Accuracy by Probit : ', num2str(acc_p)]);
plot(m, 'r+');
title('Probit');










