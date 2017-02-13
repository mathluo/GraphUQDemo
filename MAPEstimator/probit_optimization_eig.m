function [ u, iter_stats ] = probit_optimization_eig(V, E, u0, dt, gamma, max_iter, fid)
% use forward backward method for optimization
% J(u) = 1/2<u, Pu> - log likelihood(u, y)
% v = u - dt * gradlikelihood(gamma; yu); 
% u_new = 1/(1+dt\lambda_i) <v, e_i>; 
% (Kind of works, but needs to initialize u0 small(0.1), and dt small, beta
%   not too small.)

u = u0;
N = size(V, 1);
u_rec=zeros(N,max_iter);
grad_l = zeros(N, 1);
for i = 1:max_iter
    i
    % compute the gradient
    u_rec(:,i) = u;
    for k = 1:size(fid{1},1)
        grad_l(fid{1}(k)) = probit_log_likelihood_gradient(u(fid{1}(k)), 1, gamma);
    end
    for k = 1:size(fid{2},1)
        grad_l(fid{2}(k)) = probit_log_likelihood_gradient(u(fid{2}(k)), -1, gamma);
    end
    v = u + dt * grad_l; 
    u = V*((V'*v)./(ones(size(E)) + dt*E)); 
end
iter_stats.u_rec = u_rec; 
end