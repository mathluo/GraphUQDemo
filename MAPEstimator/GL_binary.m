function [ u_new,iter_stats ]  = GL_binary(V, E, u0, eps, eta, max_iter, dt, fid, varargin)
% function for Ginzburg-Landau Minimization
if(numel(varargin) == 1)
    opt = varargin{1}; 
else
    opt = {}; 
end
opt = check_opt(opt); 
% initialization
N = size(V,1);
energy=zeros(max_iter,1); 
energy_lap=zeros(max_iter,1);
energy_fid=zeros(max_iter,1);
energy_well=zeros(max_iter,1);

%u_old = ground_truth; 
u_old = u0; 

for k = 1:max_iter
   v = V*((V'*u_old)./(ones(size(E)) + eps*dt*E)); 
   v = l2_fidelity_binary(v,eta*dt,fid); 
   u_new = threshold_double_well(v,dt); 
   if opt.ZeroMean
       u_new = u_new - mean(u_new); 
   end
   % compute iteration stats
   u_hat = V'*u_new;
   energy_lap(k) = .5* sum(E.*u_hat.*u_hat);
   energy_well(k) = double_well_energy(u_new,eps);
   energy_fid(k) = fid_energy(u_new,fid,eta);
   energy(k) = energy_lap(k) + energy_well(k) + energy_fid(k);
   u_old = u_new; 
end
iter_stats.energy = energy; 
iter_stats.energy_lap = energy_lap; 
iter_stats.energy_well = energy_well; 
iter_stats.energy_fid = energy_fid; 
end


function e = double_well_energy(u,eps)
    e = .25/eps*sum((u.^2-1).^2); 
end

function e = fid_energy(u,fid,eta)
    e = .5*eta*sum((u(fid{1}) - 1).^2); 
    e = e + .5*eta*sum((u(fid{2}) + 1).^2); 
end


function u = threshold_double_well(u_old, dt)
    u = u_old - dt*(u_old.^3 - u_old); 
end

function u = l2_fidelity_binary(u_old, dt, fid_ind)
    u_old(fid_ind{1}) = u_old(fid_ind{1}) -dt * (u_old(fid_ind{1}) - 1); 
    u_old(fid_ind{2}) = u_old(fid_ind{2}) -dt * (u_old(fid_ind{2}) + 1); 
    u = u_old; 
end


function opt = check_opt(opt)
if(~isfield(opt,'ZeroMean'))
    opt.ZeroMean = false;
end
end
