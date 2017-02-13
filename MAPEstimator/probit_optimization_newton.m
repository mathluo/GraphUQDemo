function [ u, iter_stats ] = probit_optimization_newton(P, u0, gamma, max_iter, fid)
% exact optimization for probit using Newton's method.
% energy is 1/2<u, Pu> - log likelihood(u, y)
% log likelihood = \sum_{j\in Z} log(normcdf(u(j), gamma))
% Newton's method: x(n+1) = x(n) - H^-1 grad(x(n))

u = u0;
N = size(P, 1);
u_rec=zeros(N,max_iter);
grad_l = zeros(N, 1);
hess_l = zeros(N,1);
for i = 1:max_iter
    % compute the gradient
    u_rec(:,i) = u;
    for k = 1:size(fid{1},1)
        grad_l(fid{1}(k)) = probit_log_likelihood_gradient(u(fid{1}(k)), 1, gamma);
    end
    for k = 1:size(fid{2},1)
        grad_l(fid{2}(k)) = probit_log_likelihood_gradient(u(fid{2}(k)), -1, gamma);
    end
    grad_u = P*u - grad_l;
    % compute the Hessian
    for k = 1:size(fid{1},1)
        hess_l(fid{1}(k)) = probit_log_likelihood_hessian(u(fid{1}(k)), 1, gamma);
    end
    for k = 1:size(fid{2},1)
        hess_l(fid{2}(k)) = probit_log_likelihood_hessian(u(fid{2}(k)), -1, gamma);
    end
    H = P - diag(hess_l);
    u = u - H\grad_u;
end
iter_stats.u_rec = u_rec; 


end

