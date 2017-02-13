function res = probit_log_likelihood_gradient(x,y,gamma)
if numel(x) == 1
    res = component(x,y,gamma); 
else
    res = zeros(size(x));
    for i = 1:numel(x)
        res(i) = component(x(i), y(i), gamma); 
    end
end
end



function res = component(x,y,gamma)
div=sqrt(2.0*pi*gamma*gamma);
res=y*exp(-x*x/(2.0*gamma*gamma))/div;
res=res/normcdf(y*x,0,gamma);
end