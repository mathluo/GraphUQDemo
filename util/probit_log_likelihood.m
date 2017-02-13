function res = probit_log_likelihood(x, y, gamma)
if numel(x) == 1
    res = log(normcdf(x*y,0,gamma));
else
    res = zeros(size(x));
    for i = 1:numel(x)
        res(i) = log(normcdf(x(i)*y(i),0,gamma));
    end
end
end
