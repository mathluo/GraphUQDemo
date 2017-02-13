function [ Vout, Eout ] = plume_data( varargin )

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
%       num_eig : number of eigenvectors of the image.
%
%  Output :
%       V : (N1 x N2, num_eig, 7) array,
%           Eigenvector of the plume data. There are a total of 7 images
%           whose pixels are aggregated into a single dataset.
%       E : (num_eig, 1) array,
%           Eigenvalues for plume data.
% -------------------------------------------------------------------------
load('MatFiles/plume_eig.mat');
if nargin == 1
    num_eig = varargin{1};
    Vout = V(:,1:num_eig);
    Eout = E( 1:num_eig);
else
    Vout = V;
    Eout = E;
end
clear V;
clear E;

end

