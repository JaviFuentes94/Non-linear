function [y,indx_array]=knn_regress_demo(x,t,K,xtest,alpha)
% nearest neighbor regression 
% x is N,D array of inputs
% t i N,1  targets
% y output estimates at xtest
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 02457 (c) 2007 LKH, IMM, DTU
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[N,D]=size(x);
[Ntest,D]=size(xtest);
indx_array=zeros(Ntest,K);

%Add one dimension to the train and test sets
X_train=ones(N,D+1);
X_train(:,1:D)=x;
X_test=ones(Ntest,D+1);
X_test(:,1:D)=xtest;

%For each point in the test set 
y=zeros(Ntest,1);
for j=1:Ntest,
    %if rem(j,20)==0, disp([' n = ',int2str(j),' of ',int2str(Ntest)]),end
    %Calculate the distance between the test point and all the other
    %points in the training set
    delta=(x-repmat(xtest(j,:),N,1));
    dist=sum(delta.*delta,2);
    %Sort the distances so that we can select the K nearest
    [dummy indx]=sort(dist);
    X=X_train(indx(1:K),:);
    %From exercise 3 w=((X'*X)^-1)*X'*t
    w=inv(X'*X+alpha*eye(D+1))*X'*t(indx(1:K));
    %w=inv(X'*X)*X'*t(indx(1:K));
    %w=pinv(X)*t(indx(1:K));
    %Calculate the test targets
    y(j)=X_test(j,:)*w;
    indx_array(j,:)=indx(1:K)';
end,    
