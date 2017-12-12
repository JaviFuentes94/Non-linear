function [gplog_test,pred_test_t,std_pred_test_t]=gp_loglik(test_dist,test_t,test_train_dist,train_dist,train_t,sig2,beta)
%

train_N=size(train_dist,1);
test_N=size(test_dist,1);
%Using the notation from the exercise:
%C_train
A=exp(-train_dist/(2*sig2));%+(1/beta)*eye(train_N);
%C_test
B=exp(-test_dist/(2*sig2));%+(1/beta)*eye(test_N);
%C_test,train
C=exp(-test_train_dist/(2*sig2));
%C_test,train*C_train^-1
Q=C*pinv(A);
%C_test|train
Btt=B-Q*C';
%nu_test|train
pred_test_t=Q*train_t;
%Eq 6.69 from Bishop. log likelihood function of the test set ln p(t_test|hyperparameters)
gplog_test=-0.5*log(det(Btt)) -0.5*(test_t-pred_test_t)'*pinv(Btt)*(test_t-pred_test_t);
%Eq 11 exercise
std_pred_test_t=sqrt(abs(diag(Btt)));







