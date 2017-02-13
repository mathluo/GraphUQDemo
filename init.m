%--------------------------------------------------------------------------
% Author: Xiyang Luo <xylmath@gmail.com> , UCLA 
%
% This file is part of the diffuse-interface graph algorithm code. 
% There are currently no licenses. 
%
%--------------------------------------------------------------------------
% Description: Add relevant paths to the Matlab Path
%
%--------------------------------------------------------------------------

initial_path = pwd; 
addpath(genpath(strcat(initial_path,'/util')));
addpath(genpath(strcat(initial_path,'/data')));
addpath(genpath(strcat(initial_path,'/MCMC')));
addpath(genpath(strcat(initial_path,'/MAPEstimator')));
addpath(genpath(strcat(initial_path,'/Demo')));