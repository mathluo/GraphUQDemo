function [ m,iter_stats] = sgf_probit(gamma,max_iter,CH,C1,fid,ground_truth)
% stochastic gradient flow of probit

% gamma=0.1;
% max_iter=10^4; % number of time steps


N = size(CH,1);
%%%%%%%%%%% How are these stepsizes chosen??? %%%%%%%
g3=gamma*gamma; div=sqrt(2.0*pi*g3); % obs noise
c1=1/g3; % convex
deltat=2/c1; % it use to be .1/c1
sdeltat=sqrt(2) *sqrt(deltat); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mult=1.0/(1.0+deltat);

m=zeros(N,1); 
burn_in_step = 4000; 
rec_step = max_iter-burn_in_step; 
u_rec=zeros(N,rec_step);
cc=zeros(N,1);
y_mean_rec=zeros(N,rec_step); 
pointwise_err_rec = zeros(N,rec_step); 
ratio = zeros(rec_step,1); 


%random initialization
u=sign(randn(N,1));
for k=1:max_iter
    d=zeros(N,1);    
    for i = 1:size(fid{1},1)
        d(fid{1}(i)) = grad_probit(u(fid{1}(i)),1, g3, div, gamma);
    end
    for i = 1:size(fid{1},1)
        d(fid{2}(i)) = grad_probit(u(fid{1}(i)),-1, g3, div, gamma);
    end
    addn=CH*randn(N,1);
    u=mult*(u+deltat*C1*d+sdeltat*addn);
    % record only after the initial burn_in period. 
    if(k>burn_in_step)
        ind = k-burn_in_step; 
        m=((ind-1)*m+sign(u))/ind;
        u_rec(:,ind)=u;
        y_mean_rec(:,ind)=m;
        diff1 = (sign(u_rec(:,ind)) ~= ground_truth);
        ratio(ind) = 1-sum(diff1)/N;
        cc=((ind-1)*cc+diff1)/ind;
        pointwise_err_rec(:,ind)=cc;
    end
end
iter_stats.u_rec = u_rec; 
iter_stats.y_mean_rec = y_mean_rec; 
iter_stats.pointwise_err_rec = pointwise_err_rec; 
iter_stats.ratio = ratio; 
end

function d = grad_probit(x,y,g3,div,gamma)
d=y*exp(-x*x/(2.0*g3))/div;
d=d/normcdf(x,0,gamma);
end

