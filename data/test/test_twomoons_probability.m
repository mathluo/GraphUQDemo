%% Generate two moons data
disp('Generating Graph and Data...'); 
tic; 
data_params.n = 2000; 
data_params.sigma = sqrt(0.04); 
data_params.type = 'two_moons'; 
data = synthetic_data(data_params)'; 
N = size(data,1); 
disp(['Number of points: ', num2str(data_params.n)]); 

p1 = twomoons_probability(data, [0,0], 1, 0, pi, data_params.sigma); 
p2 = twomoons_probability(data, [1,0.5], 1, pi, 2*pi, data_params.sigma); 

figure; 
plot(p1); 

figure; 
plot(p2); 

q1 = p1./(p1 + p2); 
q2 = p2./(p1 + p2); 
ind = abs(q1-q2) > 0.6; 

figure; 
hold on; 
plot(data(p1>p2&ind, 1), data(p1>p2&ind, 2), 'r.'); 
plot(data(p1<p2&ind, 1), data(p1<p2&ind, 2), 'b.'); 
plot(data(abs(q1-q2) <= 0.6, 1), data(abs(q1-q2) <= 0.6, 2), 'g.'); 
hold off; 