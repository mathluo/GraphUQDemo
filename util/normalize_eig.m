function [E] = normalize_eig(S,V)
%--------------------------------------------------------------------------
%  Description : 
%       Produce the normalized precision and square_root of precision
%       matrix from an eigendecomposition of the graph Laplacian [S,V] =
%       eigs(L). 
%       Scale to make mean variance of $u$ over all nodes 1. 
%  Input : 
%       S: Neig x 1 Array. Eigenvalues of the Laplacian
%       V: N x Neig Array, cols are eigenvectors of Laplacian
%
%  Output : 
%       E: rescaled eigenvalues
% -------------------------------------------------------------------------
N = size(V,1); 
Neig = size(V,2); 
S = reshape(S,Neig,1); 
S(1)=1; si=1./S; si(1)=0.0;  SI=diag(si); 
C1=V*SI*V';
scale=N/trace(C1);
E = S/scale; 
end