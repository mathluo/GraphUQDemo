function [D] = cosdist(X, Y)
%--------------------------------------------------------------------------
% Author: Xiyang Luo <xylmath@gmail.com> , UCLA 
%
% This file is part of the diffuse-interface graph algorithm code. 
% There are currently no licenses. 
%
%--------------------------------------------------------------------------
% Description: Computes pairwise cosine distance
%
% Usage: 
%       X = (x1,x2,\dots, xm) m col vectors
%       Y = (y1,y2,\dots, yn) n col vectors
%       D_ij = <xi,yj>/|x_i||y_j|
%--------------------------------------------------------------------------
 
Yt = Y';  
XX = sqrt(sum(X.*X,1));        
YY = sqrt(sum(Yt.*Yt,2));   
XX(XX == 0) = .01;
YY(YY ==0 ) = .01; %division by 0
D = bsxfun(@rdivide,Yt,YY) * bsxfun(@rdivide,X,XX);

end

