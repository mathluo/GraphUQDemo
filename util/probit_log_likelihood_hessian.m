function res = probit_log_likelihood_hessian(x,y,gamma)
if numel(x) == 1
    res = component(x, y, gamma); 
else
    res = zeros(size(x));
    for i = 1:numel(x)
        res(i) = component(x(i), y(i), gamma); 
    end
end
end


function res = component(x, y, gamma)
    div=2.0*pi*gamma^2;
    f = normcdf(y*x, 0, gamma); 
    g = (exp(-x^2/(2.0*gamma^2))); 
    res = -f * y*x*g/(gamma^2 * sqrt(div)) - g*g/div; 
    res = res/(f^2);
end