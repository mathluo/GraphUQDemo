function [ u_out,stats ] = spectral_binary(V, ground_truth, mode)
% V: Eigenvectors
% mode: 'threshold'(just threshold the second eigenvector), 
%       'kmeans'(performs kmeans on the row vectors of V
% u_out: (-1, 1) label class prediction

V = V(:, 2:end); %discard the first eigenvector as it should be all zeros
if(strcmp(mode, 'kmeans'))
    u_out  = kmeans(V, 2); 
    u_out(u_out == 1) = 1; %remap
    u_out(u_out == 2) = -1; 
end

if(strcmp(mode, 'threshold'))
    u_out = sign(V(:, 1)); 
end

if ground_truth
    N = size(V,1); 
    diff1 = (u_out ~= ground_truth);
    err = 1-sum(diff1)/N;
    if(err > 0.5)
        err = 1 - err; 
    end
    stats.error = err; 
end

end

