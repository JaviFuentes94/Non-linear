warning off

%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mu=[5 2]';	   % true mean value
SIGMA=[1   1.7 
       1.7 5  ];   % true covariance matrix
N=10000;           % number of points in density

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nSamples=100;
Errors=zeros(nSamples,1);
for idx=5:nSamples

  N=idx;
  disp('The true mean is');
  mu
  disp('and the true covariance matrix is');
  SIGMA
  disp('The number of samples is');
  N
  
  D = randmvn(mu,SIGMA,N);  % A random sample - set of N samples.
  
  mu_ = mean(D,2);  % Estimated mean
  SIGMA_ = cov(D'); % Estimated covariance matrix

  disp('The estimated mean is');
  mu_
  disp('and the estimated covariance matrix is');
  SIGMA_
  
  fprintf('Calculating histogram ... ');
  nbins = 20;                   % # bins in histogram in each dimension
  [n,x] = hist2d(D,nbins,[],1);     % Normalized histogram
  fprintf('Done\n\r');
  resol = 100;                   % # points in plot
  range = x(:,nbins)-x(:,1);
  x1 = x(1,1):range(1)/(resol-1):x(1,nbins);
  x2 = x(2,1):range(2)/(resol-1):x(2,nbins);
  
  [X1 X2] = meshgrid(x1,x2);
  X1=X1(:);X2=X2(:);
  fprintf('Calculating True PDF ... ');
  p = mvnpdf(mu,SIGMA,[X1';X2']);% True PDF 
  p = reshape(p,[resol resol]);
  fprintf('Done\n\r');
  
%%%%%%%%%%%%%%%%%%  Plotting %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   figure(1)
%   clf
%   h0 = get(0,'defaultfigureposition');
%   h=get(gcf,'Position');
%   h(3:4)=[1.8 .8].*h0(3:4);
%   set(gcf,'Position',h);
%   set(gcf,'PaperUnits','centimeters');
%   set(gcf,'PaperType','a4');
%   set(gcf,'PaperOrientation','landscape');
%   set(gcf,'PaperPosition',[1 5 28 10]);
%   subplot 131
%     plot(D(1,:),D(2,:),'.'); 
%     ax=axis;
%     axis equal;
%     axis(ax);
%     xlabel('x_1')
%     ylabel('x_2')  
%     title('Scatter plot of data-set');
%   subplot 132
%     surf(x1,x2,p);
%     shading flat;
%     xlim([x1(1) x1(resol)])
%     ylim([x2(1) x2(resol)])  
%     ax = axis;  
%     xlabel('x_1')
%     ylabel('x_2')  
%     zlabel('p(x_1,x_2)')
%     view([-20 30])
%     title('True PDF')  
%     tick1 = get(gca,'yticklabel');
%   subplot 133
%     h= bar3xy(x(1,:),fliplr(x(2,:)),n,1,'w');
%     axis(ax);
%     set(gca,'yticklabel',flipud(tick1));
%     view([-20 30])
%     xlabel('x_1')
%     ylabel('x_2')  
%     zlabel('$\tilde{n}$','Interpreter','latex')
%     title('Normalized Histogram')  
%   
%     %%%%%% Aditional plot
%     
%     p2 = mvnpdf(mu_,SIGMA_,[X1';X2']);% True PDF 
%     p2 = reshape(p2,[resol resol]);
%     
%     figure
%     subplot(2,1,1)
%     surf(x1,x2,p2);
%     shading flat;
%     xlim([x1(1) x1(resol)])
%     ylim([x2(1) x2(resol)])  
%     ax = axis;  
%     xlabel('x_1')
%     ylabel('x_2')  
%     zlabel('p(x_1,x_2)')
%     view([-20 30])
%     title('Estimated PDF')  
%     tick1 = get(gca,'yticklabel');
%     
%     subplot(2,1,2)
%     surf(x1,x2,p);
%     shading flat;
%     xlim([x1(1) x1(resol)])
%     ylim([x2(1) x2(resol)])  
%     ax = axis;  
%     xlabel('x_1')
%     ylabel('x_2')  
%     zlabel('p(x_1,x_2)')
%     view([-20 30])
%     title('True PDF')  
%     tick1 = get(gca,'yticklabel');
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    p2 = mvnpdf(mu_,SIGMA_,[X1';X2']);% EStimated PDF 
    p2 = reshape(p2,[resol resol]);
    
    error=(p-p2).^2;
    
    Errors(idx)=sum(sum(error))/resol^2;
  
end

figure
subplot(1,2,1)
loglog([1:nSamples]*100,Errors)
xlabel('log(N)')
ylabel('log(Error)')
subplot(1,2,2)
plot([1:nSamples]*100,Errors)
xlabel('N')
ylabel('Error')


