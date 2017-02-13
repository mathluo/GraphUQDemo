function [ uout ] = double_well_threshold(uin, eps, T)
% solve numerically the ODE equation using forward stepping. 
dt = 0.003; 
n = ceil(T/dt); 
for i = 1:n
   uout = uin - dt/eps * (uin.^3 - uin); 
   uin = uout; 
end


end

