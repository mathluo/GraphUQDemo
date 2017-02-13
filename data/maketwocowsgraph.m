% construct graphs for for two cows using method in Flenner's paper. 
im = imread('originalCow.png'); 
% define feature constants
nbsz = 5; 
tau = 4; 

% fidelity points
p1 = [111, 84]; %black cow
p2 = [124, 96]; 
temp = zeros(size(im, 1), size(im, 2)); 
for i = p1(2):p2(2)
    for j = p1(1) :p2(1)
        temp(i,j) = 1; 
    end
end
ind1 = find(reshape(temp, numel(temp), 1) == 1); 

p1 = [182, 135]; %red cow
p2 = [204, 146]; 
temp = zeros(size(im, 1), size(im, 2)); 
for i = p1(2):p2(2)
    for j = p1(1) :p2(1)
        temp(i,j) = 1; 
    end
end
ind2 = find(reshape(temp, numel(temp), 1) == 1); 


p1 = [156, 17]; %sky
p2 = [178, 28]; 
temp = zeros(size(im, 1), size(im, 2)); 
for i = p1(2):p2(2)
    for j = p1(1) :p2(1)
        temp(i,j) = 1; 
    end
end
ind3 = find(reshape(temp, numel(temp), 1) == 1); 


p1 = [53, 162]; %grass
p2 = [83, 177]; 
temp = zeros(size(im, 1), size(im, 2)); 
for i = p1(2):p2(2)
    for j = p1(1) :p2(1)
        temp(i,j) = 1; 
    end
end
ind4 = find(reshape(temp, numel(temp), 1) == 1); 


% read cow image and extract convolution features
im = double(im); 
im = im / 255; 
opt = {}; 
opt.output_dim = 2; 
opt.kernel = true; 
data = image2patches(im, nbsz, opt); 

% use nystrom
opt = {}; 
opt.tau = tau; 
opt.Metric = 'Euclidean'; 
opt.Laplacian = 'n'; 
opt.numsample = 500; 
opt.neig = 45; 
[V, E] = nystrom(data, opt); 

% plot eigenvectors and eigenvalues
figure; 
plot(E); 
figure; 
imagesc(reshape(V(:, 2), size(im, 1), size(im, 2))); 
figure; 
imagesc(reshape(V(:, 3),  size(im, 1), size(im, 2))); 
figure; 
imagesc(reshape(V(:, 4),  size(im, 1), size(im, 2))); 
figure; 
imagesc(reshape(V(:, 5),  size(im, 1), size(im, 2))); 
figure; 
imagesc(reshape(V(:, 6),  size(im, 1), size(im, 2))); 

labels = {}; 
labels.blackcow = ind1; 
labels.redcow = ind2; 
labels.sky = ind3; 
labels.grass = ind4; 
imsz = [size(im, 1), size(im, 2)]; 
save('MatFiles/twocowsgraph.mat', 'V', 'E', 'labels', 'imsz'); 






