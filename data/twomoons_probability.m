function [p] = twomoons_probability(data, center, radius, theta1, theta2, sigma )
% function to calculate the joint probability distribution of two moons
% data : (N x k) vector, N being number of data points, 
% theta1 : starting angle
% theta2 : end angle
% radius : radius of the arc
% sigma : std deviation of noise. 

k = (theta2-theta1); 
p = zeros(size(data,1),1); 
c1 = center(1); 
c2 = center(2); 

for i = 1:size(data,1)
    x = data(i,1); 
    y = data(i,2); 
    f = @(t)integrant(radius, c1, c2, theta1, k, x, y, sigma, t);
    p(i) = integral(f, 0, 1); 
end

end


function res = integrant(r, c1, c2, theta1, k, x, y, sigma, t)
    sqd = (c1 + r* cos(theta1 + k*t) - x).*(c1 + r* cos(theta1 + k*t) - x)+ (c2 + r* sin(theta1 + k*t) - y).*(c2 + r* sin(theta1 + k*t) - y); 
    res = exp(-sqd/(2*sigma*sigma))/(2*pi*sigma*sigma); 
end
