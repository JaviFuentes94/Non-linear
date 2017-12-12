% COURSE 02457 EXERCISE 11 
% Lars Kai Hansen (c) 2001
%
% MARKOV CHAIN ESTIMATION'
%
clear, close all
% Initialization
K=10;  % number of states
%initialize transition probabilities of
% the "teacher" model
a=rand(K);
amean=sum(a,2);
anorm=a.*repmat(1./amean,1,K);

% The eigenvector corresponding to (maximal) unit eigenvalue 
% is the stationary probability distribution, find it by iteration
p=zeros(1,K);
p(1)=1
for j=1:100,
    p=p*anorm;
end
p_stat=p;

% Get random sequence from the
% teacher model

% compute the cumulative distribution
ac=anorm;
for j=1:K,
    ac(j,:)=cumsum(anorm(j,:));
end
n=1;
Nmax=10000;
% get  integers from the distribution
error_vec = [];
for j=1:Nmax,
   n=getint(ac(n,:));
   h(j)=n;
   bins=hist(h,K);
   bins = bins/j;
   error1 = sum((p - bins)*(p - bins)');
   error_vec = [error_vec error1];
   if mod(j,500) == 0
        disp('Iteration')
        figure(1),
        subplot(2,1,1),bar(p),
        title('TRUE STATIONARY DIST')
        subplot(2,1,2)
        bar(bins)
        title('ESTIMATED STATIONARY DIST')
        drawnow
   end
%     figure(1),
%     % plot the true stationary distribution
%     subplot(2,1,1),bar(p),
%     title('TRUE STATIONARY DIST')
%     % plot the histogram of occurrence in the sample
%     subplot(2,1,2)
%     bins=hist(h,K);
%     bar(bins/j)
%     title('ESTIMATED STATIONARY DIST')
%     drawnow
%    end
end
%
figure(2);
t = 50:4000
plot(t, error_vec(50:4000))
title('Error between stationary distribution and observed dist')
xlabel('Sample size'),
ylabel('Error'),

% 
% Estimate Markov model from sequences of
% increasing length to plot "learning curve"
% show convergence of Markov estimation
N_sizes=100; 
for n=1:N_sizes,
    Nmax=n*1000;  % sequence length
    narray(n)=Nmax;
    nn=1;
    for j=1:Nmax,
      nn=getint(ac(nn,:));
      seq(j)=nn;
    end
    aest=markov_map(seq,K,eps);
    error(n)=mean(mean(((aest-anorm).^2)./(anorm.^2)));
    figure(3)
    plot(narray(1:n),error(1:n)),
    xlabel('SAMPLE SIZE'),
    ylabel('SQUARE ERROR IN TRANSITION MATRIX'),
    drawnow
end
