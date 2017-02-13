function res = probit_data_log_likelihood_grad(u, ind, y, gamma)
res = zeros(size(u)); 
for i = 1:numel(ind)
    res(ind(i)) = component(u(ind(i)), y, gamma); 
end
function res = component(x,y,gamma)
div=sqrt(2.0*pi*gamma*gamma);
res=y*exp(-x*x/(2.0*gamma*gamma))/div;
res=res/normcdf(y*x,0,gamma);
end
end
