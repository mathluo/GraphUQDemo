function [phi, E] = nystrom(data, opt)

%--------------------------------------------------------------------------
% Author: Xiyang Luo <xylmath@gmail.com> , UCLA 
%
% This file is part of the diffuse-interface graph algorithm code. 
% There are currently no licenses. 
%
%--------------------------------------------------------------------------
%  Description : Nystrom Extension for Normalized Laplacian
%
%  Input : 
%       data = (x1T;x2T;...xmT), where individual points are rows
%       opt. tau : kernel width for metric
%       opt. numsample : number of samples
%       opt. neig  : number of eigenvectors
%       opt. Metric : 'Euclidean' or 'Cosine'
%       opt. Laplacian = 'n' or 'u'
%
%  Output : 
%       phi : (phi_1,...phi_n), coloumn eigenvectors
%       E   : Array of eigenvalues (increasing)
% -------------------------------------------------------------------------

opt = check_opt(opt); 
tau = opt.tau;
num_samples = opt.numsample;
neig = opt.neig;

% randomly select samples
num_rows = size(data, 1);
permed_index = randperm(num_rows);
sample_data = data(permed_index(1:num_samples), :);
other_data = data(permed_index(num_samples+1:num_rows), :, :);
clear data;

% calculate the weights distances 
other_points = num_rows - num_samples;
A = zeros(num_samples, num_samples);
B = zeros( num_samples, other_points);

if strcmp(opt.Metric, 'Cosine') % sketchy
    foo = cosdist(sample_data',sample_data');
    foo(foo<=0) = 0; %hard thresholding negative values
    A = exp(-abs((1./(foo+.01)-.9901).^(1.3))/tau); 
    A(1:size(A,1)+1:size(A,1)*size(A,2)) = 0;
    foo = cosdist(other_data',sample_data');
    foo(foo<=0) = 0; %hard thresholding negative values
    B = exp(-abs((1./(foo+.01)-.9901).^(1.3))/tau);     
    clear foo;
end

if strcmp(opt.Metric, 'Euclidean')
    A = sqdist(sample_data',sample_data');   
    if(opt.tau == -1) % automatic
        auto_tau = prctile(A(:),18); 
        tau = auto_tau*1.1; 
    end
    B = sqdist(other_data',sample_data'); 
    A = exp(-A/tau);
    B = exp(-B/tau);
    A(1:size(A,1)+1:size(A,1)*size(A,2)) = 0; %set diagonal to 0 
end

clear sample_data other_data;


% Normalize A and B using row sums of W, where W = [A B; B' B'*A^-1*B].
% Let d1 = [A B]*1, d2 = [B' B'*A^-1*B]*1, dhat = sqrt(1./[d1; d2]).

if opt.Laplacian == 'n'
    B_T = B';
    d1 = sum(A, 2) + sum(B, 2);
    d2 = sum(B_T, 2) + B_T*(pinv(A)*sum(B, 2));
    dhat = sqrt(1./[d1; d2]);
    A = A .* (dhat(1:num_samples)*dhat(1:num_samples)');
    B1 = dhat(1:num_samples)*dhat(num_samples+(1:other_points))';
    B = B .* B1;
    clear d1 d2 B1 dhat;
end


% Do orthogalization and eigendecomposition
Asi = sqrtm(pinv(A));
B_T = B';
BBT = B*B_T;
W = single(zeros(size(A, 1)+size(B_T, 1), size(A, 2)));
W(1:size(A, 1), :) = A;
W(size(A, 1)+1:size(W, 1), :) = B_T;
clear B B_T;
% Calculate R = A + A^-1/2*B*B'*A^-1/2
R = A + Asi*BBT*Asi;
R = (R + R')/2; % Make sure R is symmetric, sometimes R can be non-symmetric because of numerical inaccuracy
[U,E] = eig(R);
[~, ind] = sort(diag(E), 'descend');
U = U(:, ind); % in decreasing order
E = E(ind, ind); % in decreasing order
E = diag(E);
clear A R BBT;
W = W*Asi;
phi = bsxfun(@rdivide, W*U, sqrt(E'));

phi(permed_index,:) = phi;
phi = real(phi);
E = 1-E;

phi = phi(:,1:neig); 
E = E(1:neig);
end



function opt = check_opt(opt)
if(~isfield(opt, 'tau'))
    opt.tau = 1; 
end
if(~isfield(opt,'Laplacian'))
    opt.Laplacian = 'n'; 
end
if(~isfield(opt,'Metric'))
    opt.Metric= 'Euclidean'; 
end
if(~isfield(opt,'numsample'))
    opt.numsample = 200; 
end
if(~isfield(opt,'neig'))
    opt.neig = 50; 
end
end




