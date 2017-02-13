function [ u_new,iter_stats ]  = MBO_binary(V, E, u0, eta, max_iter, inner_iter, dt, fid, ground_truth)
% function for MBO iteration.

% initialization
N = size(V,1);
u_rec=zeros(N,max_iter);

%u_old = ground_truth;
u_old = u0;

for k = 1:max_iter
    v = u_old; 
    for s = 1:inner_iter
        v = V*((V'*v)./(ones(size(E)) + (dt)*E));
        v = l2_fidelity_binary(v,eta*(dt),fid);
    end
    u_new = threshold_binary_hard(v);
    % compute iteration stats
    u_rec(:,k) = u_new;
    diff1 = (sign(u_rec(:,k)) ~= ground_truth);
    ratio(k) = 1-sum(diff1)/N;
    u_old = u_new;
end
iter_stats.u_rec = u_rec;
iter_stats.ratio = ratio;
end


function u = threshold_binary_hard(u_old)
u_old(u_old>0) = 1;
u_old(u_old<0) = -1;
u = u_old;
end

function u = l2_fidelity_binary(u_old, dt, fid_ind)
u_old(fid_ind{1}) = u_old(fid_ind{1}) -dt * (u_old(fid_ind{1}) - 1);
u_old(fid_ind{2}) = u_old(fid_ind{2}) -dt * (u_old(fid_ind{2}) + 1);
u = u_old;
end