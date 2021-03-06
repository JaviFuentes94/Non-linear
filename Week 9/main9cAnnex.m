% MATLAC exercise main9c.m
%
D=1:100;  % input dimension of sunspot prediction 

minError = [];
for d=D
[x,tr_t,xtest,te_t] = getsun(d);
var=std([tr_t',te_t'])^2;
%
Ntrain=length(tr_t);
Kmax=Ntrain;
alpha=0.005;
Ntest=length(te_t);
for K=1:Kmax
    ypred=knn_regress_demo(x,tr_t,K,xtest,alpha);
    Error(K)=sum((ypred-te_t).^2)/(Ntest*var);
end
minError=[minError min(Error)];
end
figure(1)
hold off
plot(1:Kmax,Error,'o',1:Kmax,Error,'-')
xlabel('NEAREST NEIGHBORS K')
ylabel('TEST ERROR'),grid
axis([0 Kmax 0 1.1*max(Error)])

[dummy Kopt]=min(Error);
figure(2)
ypred=knn_regress_demo(x,tr_t,Kopt,xtest,alpha);
plot(1920+(1:Ntest),te_t,'r-',1920+(1:Ntest),ypred,'b-',1920+(1:Ntest),te_t,'ro',1920+(1:Ntest),ypred,'bo')
title(['K_{opt} = ', int2str(Kopt)])
grid, xlabel('YEAR'), ylabel('SUN SPOT INTENSITY')

figure(3)
plot(D,minError)
title('Minimum error for different dimensions of data and K=1:100')
xlabel('Number of dimensions used')
ylabel('Minimum error')