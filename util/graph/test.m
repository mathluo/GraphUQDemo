%% plot Synthetic data
% % plot the synthetic data
% figure; 
% params.type = 'two_moons'; 
% params.n = [100,1000]; 
% [x,g] = synthetic_data(params); 
% scatter(x(1,:),x(2,:),30,g); 
% 
% figure; 
% params.type = 'two_blobs'; 
% params.n = [100,900]; 
% [x,g] = synthetic_data(params); 
% scatter(x(1,:),x(2,:),30,g); 
% 
% figure; 
% params.type = 'one_blob'; 
% params.n = 1000; 
% [x,g] = synthetic_data(params); 
% scatter(x(1,:),x(2,:),30); 
% 
% figure; 
% params.type = 'two_circles'; 
% params.n = [100,900]; 
% params.r1 = 1; 
% params.r2 = 2; 
% [x,g] = synthetic_data(params); 
% scatter(x(1,:),x(2,:),30,g); 
% 
% figure; 
% params.type = 'three_circles'; 
% params.n = [100,400,900]; 
% params.r1 = 0; 
% params.r2 = 1;
% params.r3 = 2; 
% [x,g] = synthetic_data(params); 
% scatter(x(1,:),x(2,:),30,g); 


% figure; 
% params.type = 'three_sticks'; 
% params.n = [100,400,900]; 
% params.sigma = .04; 
% [x,g] = synthetic_data(params); 
% scatter(x(1,:),x(2,:),10,g); 


%% test Nystrom 
% im = imread('../grass.jpg'); 
% im = imresize(im,.5); 
% sigma = 5; 
% tic; 
% data = image2patches(im,3,struct('kernel',true,'output_dim',2)); 
% % % test kdtree eigenvectors
% % W = data2sparseweight(data,struct('num_neighbors',50,'tau',sigma)); 
% % L = sparse_laplacian(W,struct('type','s'));
% % toc; 
% % tic; 
% % [V,E] = eigs(L,12,'sr'); 
% % E = diag(E); 
% % toc; 
% % tile(V(:,1:12),struct('colormap',colormap(jet), 'xsize', size(im,1), ...
% %     'ysize', size(im,2))); 
% % figure; 
% % plot(E); 
% 
% 
% 
% %Nystrom Extension
% [V,E] = nystrom(data,struct('tau',10,'numsample',200, 'neig',12, ...
%     'Metric', 'Euclidean', 'Laplacian','n')); 
% tile(V(:,1:12),struct('colormap',colormap(jet), 'xsize', size(im,1), ...
%     'ysize', size(im,2))); 
% figure; 
% plot(E); 


%% test the graph construction by plotting spectral embedding

figure; 
params.type = 'two_moons'; 
params.n = [500,500]; 
params.sigma = .1; 
[x,g] = synthetic_data(params); 
scatter(x(1,:),x(2,:),20,g); 
dist_mat = sqdist(x,x); 
opt.graph = 'z-p'; 
opt.k = 10; 
opt.tau = 1;
opt.type = 's'; 
L = dense_laplacian(dist_mat,opt); 
[V,E] = eigs(L,10,'sr'); 
figure; 
scatter(V(:,2), V(:,3),20,g); 







    