function [ out_labels, out_images ] = mnist_data(digits)

%--------------------------------------------------------------------------
% Author: Xiyang Luo <xylmath@gmail.com> , UCLA 
%
% This file is part of the diffuse-interface graph algorithm code. 
% There are currently no licenses. 
%
%--------------------------------------------------------------------------
%  Description : Generate synthetic point cloud data
%
%  Input (params): 
%       digits: 1 x M array of digits in the label, e.g. [4, 9]    
%
%  Output : 
%       out_labels: N x 1 array, where N is the number of digits picked
%       out_images: N x 784 array, 784 = 28 x 28 being the length of the
%                   flattened array of the image of each digit. 
% -------------------------------------------------------------------------
load('MatFiles/mnist.mat');
ind = []; 
for i = 1:numel(digits)
    ind = [ind; find(labels == digits(i))]; 
end
out_labels = labels(ind); 
images = images'; 
out_images = images(ind,:); 
end

