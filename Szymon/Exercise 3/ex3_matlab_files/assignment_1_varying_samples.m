% MATLAB program for exercise 3 in course 02457
% This program is for part 1 out of 3 
%
% "main3a" illustrates the use of a linear model and discriminant
% in a single layer network.
% 
% The parameters that should be changed are
%   w_t        : the true weight-vector used to generate training-set
%   noiselevel : the Variance of Gaussian noise on training-set
%   N          : Number of points in training set (= n^2)
% 
% The number of dimensions can be changed (by changing length of w_t),
% but the result is only plotted for w_t=[w0 w1 w2]'

%   (c) Karam Sidaros, September 1999.
%  Uses 
%

%%%%%%%%%%%%%%%%%%%%%%%%% Part 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  Linear Models %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off

%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w_t = [7 3 5]';	   % True weights

noiselevel = 0.5;    % Standard deviation of Gaussian noise on data
  
err_vector = []
samples_vector = []

for num_samples=100:1:10000
    N = num_samples;           % Number of points in training set

    d = length(w_t)-1;  % Number of dimensions
    n=round(N^(1/d));  % Number of points in each dimension
    N = n^d;    
    
    x0 = -1:2/(n-1):1;
    x0 = x0';
    
    X = ones(N,1);     % X matrix on a d-dimensional meshgrid
    X(:,2) = repmat(x0,n^(d-1),1);
    
    for a = 3:d+1;
        v = repmat(X(1:n^(a-2),a-1),1,n)';
        v = v(:);
        X(:,a) = repmat(v,n^(d-a+1),1);
    end
    
    T = (X*w_t);
    noise = randn(N,1) * noiselevel;
    T = T + noise;

    w = pinv(X)*T;
    %w = inv(X'*X)*X'*T

    Y = X*w;
    err = mean((Y-T).^2);
    samples_vector = [samples_vector num_samples];
    err_vector = [err_vector err];
end

figure(1);
plot(samples_vector, err_vector);
xlabel('number of samples');
ylabel('error');