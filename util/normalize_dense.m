function [P,G1,C1] = normalize_dense(S,V)
%--------------------------------------------------------------------------
%  Description : 
%       Produce the normalized precision and square_root of precision
%       matrix from an eigendecomposition of the graph Laplacian [S,V] =
%       eig(L). 
%       Scale to make mean variance of $u$ over all nodes 1. 
%  Input : 
%       S: N x 1 Array. Eigenvalues of the Laplacian
%       V: N x N Array, cols are eigenvectors of Laplacian
%
%  Output : 
%       P: scaled Precision matrix that has high eigenvalues for the null
%       space of L. 
%       G1: the normalized square root of the covariance matrix $C$
% -------------------------------------------------------------------------

N = size(V,1); 
S(1)=1; si=1./S; si(1)=0.0;  SI=diag(si); 
C1=V*SI*V'; G1=V*diag(sqrt(si))*V';
scale=N/trace(C1);
G1=sqrt(scale)*G1; G1=0.5*(G1+G1');
C1 = scale*C1; C1 = .5*(C1+C1'); 
S(1)=scale*1; S=diag(S);
P=V*S*V'/scale; P=0.5*(P+P'); 
end