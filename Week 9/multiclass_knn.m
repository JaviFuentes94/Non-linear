function [looclass1,class2,Kopt,error_array]=multiclass_knn(input1,class1,input2,K)
%
% Use simple squared dist to compute the nearest
% neighbor classes by voting in K neighbors.
%
% INPUT:
%
%  input1, input2: input vectors for training and test set (D x N)
%  class1:  vector ( N X 1) with the class labels range 1:C
%           
%  K: the maximum number of neighbors
%  
% OUTPUT:
% 
%  Kopt is the leave-one-out optimal number of neighbors for which we then
%        compute:
%   class2          are the labels of the input2 from input1
%   looclass1       are the labels of class1 voting without it self included
%  (ie. the leave-one-out classes)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (c) Lars Kai Hansen IMM, DTU, July 2004
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C=max(class1);  % the number of classes
%
[D,N1]=size(input1);
[DD,N2]=size(input2);
if D~=DD, disp('Dimensional mismatch input1,input2'),end
%
%
input1_2=sum(input1.*input1,1);  %N1 by 1
input2_2=sum(input2.*input2,1);  % N2 by 1
%
% compute N1*N2 distance matrix. Distance between the test set and
% the training set
W21=(repmat(input1_2',1,N2)+repmat(input2_2,N1,1)-2*input1'*input2)'; %(InTrain - InTest)^2
% compute N1*N1 distance matrix. Distance between the different points of
% the training set
W11=repmat(input1_2,N1,1)+repmat(input1_2',1,N1)-2*input1'*input1;
%Sort the distances 
[dummy,index12]=sort(W21,2);
[dummy,index11]=sort(W11,2);
%Store the labels
labels=zeros(N1,C);
for c=1:C,
 indx=find(class1==c);
 labels(indx,c)=1;
end
disp('Ready to classify (LOO)')
%
%For each number of neighbours
for k=1:K, 
    %For each point in the training set
 for n=1:N1,
     %Check the class of the n nearest neighbour
   [dummy,cla]=max(sum(labels(index11(n,2:(k+1)),:),1));
   looclass1(n)=cla;
 end,
 %whos
%Calculate the error of missclassification
 error_array(k)=sum(class1~=looclass1)/N1;
end
%The optimal error and the optimal K
[Err_opt,Kopt]=min(error_array);
disp('Ready to classify (TEST)')
class2=zeros(1,N2);
looclass1=zeros(1,N1);
%Classify test by voting with Kopt from the training set
for n=1:N2,
   [dummy,cla]=max(sum(labels(index12(n,1:Kopt),:),1));
   class2(n)=cla;
end,
disp('Ready to classify (LOO Kopt)')
%We evaluate again in the training set with Kopt to get the classification
%of the training set. 
for n=1:N1,
   [dummy,cla]=max(sum(labels(index11(n,2:(Kopt+1)),:),1));
   looclass1(n)=cla;
end,
