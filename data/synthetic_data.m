function [X,ground_truth] = synthetic_data(params)

%--------------------------------------------------------------------------
% Author: Xiyang Luo <xylmath@gmail.com> , UCLA 
%
% This file is part of the diffuse-interface graph algorithm code. 
% There are currently no licenses. 
%
%--------------------------------------------------------------------------
%  Description : Generate synthetic point cloud data
%
%  Input (params): 
%           type : type of the data to generate. Currently: 'two_moons',
%                  'one_blob', 'two_blobs', 'two_circles', 'three_circles'
%           sigma : Standard deviation of the Gaussian noise. 
%           n : number of data points
%
%  Output : 
%       X : (k x N) dimensional data. N being # datapoints
%       ground_truth : label for classification. {-1,1} binary, {1,2,..k}
%                      for multiclass 
% -------------------------------------------------------------------------


params = check_opt(params); 

if(strcmp(params.type,'two_moons'))
    if(numel(params.n) == 1)
        num_pos = params.n/2;
        num_neg = params.n/2;
    else
        num_pos = params.n(1);
        num_neg = params.n(2); 
    end
    radii_pos = ones(num_pos,1);
    phi_pos = (1:1:num_pos)/num_pos*pi;
    radii_neg = ones(num_neg,1);
    phi_neg = (1:1:num_neg)/num_neg*pi;
    N = num_pos+ num_neg;
    X = zeros(2,N);
    ground_truth = zeros(1,N);
    for i = 1:num_pos
        X(1,i) = radii_pos(i)*cos(phi_pos(i));
        X(2,i) = radii_pos(i)*sin(phi_pos(i));
        ground_truth(i) = 1;
    end
    for i = 1:num_neg
        X(1,num_pos+i) = 1+radii_neg(i)*cos(phi_neg(i));
        X(2,num_pos+i) = -radii_neg(i)*sin(phi_neg(i))+0.5;
        ground_truth(i+num_pos) = -1;
    end
    X = X + randn(size(X))*params.sigma; 
end

if(strcmp(params.type,'one_blob'))
    n = params.n;
    X = randn(2,n)*params.sigma;
    ground_truth = nan; 
end

if(strcmp(params.type,'two_blobs'))
    if(numel(params.n) == 1)
        num_pos = params.n/2;
        num_neg = params.n/2;
    else
        num_pos = params.n(1);
        num_neg = params.n(2); 
    end    
    N = num_pos+ num_neg;
    X = zeros(2,N);
    ground_truth = zeros(1,N);
    X(:,1:num_pos) = bsxfun(@minus,randn(2,num_pos)*params.sigma, [0;-1]);
    X(:,num_pos+1:num_pos+num_neg) = bsxfun(@minus, randn(2,num_neg)*params.sigma,[0;1]);
    ground_truth(1:num_pos) = 1; 
    ground_truth(num_pos+1:num_pos+num_neg) = -1; 
end

if(strcmp(params.type,'two_circles'))
    if(numel(params.n) == 1)
        num_pos = params.n/2;
        num_neg = params.n/2;
    else
        num_pos = params.n(1);
        num_neg = params.n(2); 
    end    
    N = num_pos+ num_neg;
    X = zeros(2,N);
    ground_truth = zeros(1,N);
    r1 = params.r1;
    phi_pos = (1:1:num_pos)/num_pos*2*pi;
    r2 = params.r2;
    phi_neg = (1:1:num_neg)/num_neg*2*pi;
    for i = 1:num_pos
        X(1,i) = r1*cos(phi_pos(i));
        X(2,i) = r1*sin(phi_pos(i));
        ground_truth(i) = 1;
    end
    for i = num_pos+1:num_pos+num_neg
        X(1,i) = r2*cos(phi_neg(i-num_pos));
        X(2,i) = r2*sin(phi_neg(i-num_pos));
        ground_truth(i) = -1;
    end 
    X = X + randn(size(X))*params.sigma; 
end


if(strcmp(params.type,'three_circles'))
    if(numel(params.n) == 1)
        num_1 = params.n/3;
        num_2 = params.n/3;
        num_3 = params.n/3;
    else
        num_1 = params.n(1);
        num_2 = params.n(2); 
        num_3 = params.n(3); 
    end    
    N = num_1+ num_2+num_3;
    X = zeros(2,N);
    ground_truth = zeros(1,N);
    r1 = params.r1;
    phi_1 = (1:1:num_1)/num_1*2*pi;
    r2 = params.r2;
    phi_2 = (1:1:num_2)/num_2*2*pi;
    r3 = params.r3;
    phi_3 = (1:1:num_3)/num_3*2*pi;    
    for i = 1:num_1
        X(1,i) = r1*cos(phi_1(i));
        X(2,i) = r1*sin(phi_1(i));
        ground_truth(i) = 1;
    end
    for i = num_1+1:num_1+num_2
        X(1,i) = r2*cos(phi_2(i-num_1));
        X(2,i) = r2*sin(phi_2(i-num_1));
        ground_truth(i) = 2;
    end 
    for i = num_1+num_2+1:num_1+num_2+num_3
        X(1,i) = r3*cos(phi_3(i-num_1-num_2));
        X(2,i) = r3*sin(phi_3(i-num_1-num_2));
        ground_truth(i) = 3;
    end     
    X = X + randn(size(X))*params.sigma;  
end



end


function params = check_opt(params)
if(~isfield(params,'type'))
    error('Please Provide Data Type'); 
end

if(strcmp(params.type,'two_moons'))
    if(~isfield(params,'sigma'))
        params.sigma = .04; 
    end
end

if(strcmp(params.type,'two_blobs'))
    if(~isfield(params,'sigma'))
        params.sigma = .2; 
    end
    
end

if(strcmp(params.type,'one_blob'))
    if(~isfield(params,'sigma'))
        params.sigma = .2; 
    end
    
end

if(strcmp(params.type,'two_circles'))
    if(~isfield(params,'sigma'))
        params.sigma = .1; 
    end    
end


if(strcmp(params.type,'three_circles'))
    if(~isfield(params,'sigma'))
        params.sigma = .1; 
    end    
end


end