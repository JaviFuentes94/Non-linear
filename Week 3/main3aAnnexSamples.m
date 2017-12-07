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

noiselevel = 4;    % Standard deviation of Gaussian noise on data
  
No = 50:50:10000;           % Number of points in training set

vErrors=[];
for i=No
    
    N=i;
    d = length(w_t)-1;  % Number of dimensions
    n=round(N^(1/d));  % Number of points in each dimension
    N = n^d;

    w_t
    noiselevel
    N

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
    %%%%%%%%%%%%%%%%%%%%%% Results %%%%%%%%%%%%%%%%%%%
    disp('The calculated weight-vector is ');
    w
    disp('The training error is');
    err

    vErrors=[vErrors err];
        
    
end
figure(3)
title('Mean square error over number of samples')
plot(No,vErrors)
xlabel('Number of samples')
ylabel('Mean squared error')


%%%%%%%%%%%%%%%%%%% Plotting 3D %%%%%%%%%%%%%%%%%%%
