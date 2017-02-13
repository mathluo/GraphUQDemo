function [ X, ground_truth, fid ] = voting_data(varargin)

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
%       varargin :
%           varargin{1} : fidelity mode ,
%               'default', pick the default 5 members(2 R 3 D) as fidelity
%           varargin{2} : M, number of features to select. Default 16
%           varargin{3} : noise_level: corruption level: [0, 1]. 
%
%  Output :
%       fid : (2 x 1) cell array, indices for fidelity points.
%       X : (N x M) matrix, data points for voting records.
%       ground_truth : (N x 1) ground truth label, 1 and -1.
% -------------------------------------------------------------------------
load('MatFiles/weights_sorted.mat')
nVarargs = length(varargin);
noise_level = 0;
if nVarargs == 1
    fidelity = varargin{1};
    M = 16;
end
if nVarargs == 2
    fidelity = varargin{1};
    M = varargin{2};
end
if nVarargs == 3
    fidelity = varargin{1};
    M = varargin{2};
    noise_level = varargin{3};
end
N=435; % number of graph nodes
X=zeros(N,M); % preallocate space
for i=1:N
    X(i,1:M)=weights_sorted(i,2:M+1);
end
if ischar(fidelity)
    if strcmp(fidelity, 'default')
        fid{1} = [21,50, 150];
        fid{2} = [N, N-50];
    end
else 
    fid{1} = randi([1,267], ceil(267.0/N*fidelity), 1);
    fid{2} = randi([268,N], ceil((1.0 - 267.0/N)*fidelity), 1); 
end


ground_truth = zeros(N,1);
ground_truth(1:267) = 1;
ground_truth(268:N) = -1;
if noise_level ~= 0
    Xnew = zeros(size(X)); 
    p = 1 - 0.5 * noise_level; 
    for i = 1:size(X, 1)
        for j = 1:size(X,2)
            if rand < p
                flp = 1;
            else
                flp = -1; 
            end
            Xnew(i,j) = X(i,j) * flp; 
        end 
    end
    X = Xnew;       
end
clear weights_sorted;
end

