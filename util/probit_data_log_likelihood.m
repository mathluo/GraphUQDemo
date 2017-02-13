function res = probit_data_log_likelihood(u, ind, y, gamma)
res = 0; 
for i = 1:numel(ind)
    res = res + log(normcdf(u(ind(i)) * y,0,gamma));
end
end

