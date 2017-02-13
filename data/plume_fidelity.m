function [ fid ] = plume_fidelity( fid_perc )

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
%       fid_perc: percentage of fidelity points(0-1).
%
%  Output :
%       fid{1}: labels for plume. 
%       fid{2}: labels for other points.
%       |fid{1}| = |fid{2}|. 
% -------------------------------------------------------------------------
load('MatFiles/plume_labels1.mat');
l1 = labels.plume; 
l2 = [labels.horizon; labels.sky; labels.ground]; 
N = numel(l1); 
N = ceil(N * fid_perc); 
N1 = numel(l1); 
N2 = numel(l2); 
ind1 = randperm(N1); 
ind2 = randperm(N2); 
fid = {}; 
fid{1} = l1(ind1(1:N)); 
fid{2} = l2(ind2(1:N)); 
end

