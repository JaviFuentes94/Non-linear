% MATLAB program for exercise 3 in course 02457
% This program is for part 2 out of 3 
%
% "main3b" illustrates the use of a linear model in a single 
% layer network to model the number of sunspots.
% 
% The parameters that should be changed are
%   d : The number of dimensions of the training-set.

%   (c) Karam Sidaros, September 1999.
%  Uses 
%


%%%%%%%%%%%%%%%%%%%%%%%%% Part 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  Linear Models %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off

%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d = 15;             % Number of dimensions
S = load('sp.dat'); % Load sunspot data-set
%S = S(1:50,1:2) %Cutting the number of years in the dataset
year = S(:,1);  
S = S(:,2);
Sor=S;
d
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
error_vector = [];
d_vector = [];
for iterator=1:280
    d = iterator;
    N = length(S)-d;
    S(1:d)=randn(1,d);
    T = S(d+1:length(S));
    X = ones(N,1);
    for a = 1:N
    X(a,2:d+1) = S(a:a+d-1)';
    end

    w = pinv(X)*T;
    %w = inv(X'*X)*X'*T

    Y = X*w;
    err = mean((Y-T).^2);
    
    error_vector = [error_vector err];
    d_vector = [d_vector iterator];
end

plot(d_vector, error_vector)
xlabel('Number of years d')
ylabel('Mean squared error')