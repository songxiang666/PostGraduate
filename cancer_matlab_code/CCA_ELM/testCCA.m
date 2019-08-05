clc,clear
load('hearttest.mat')
X=hearttest(:,1:10);
Y=hearttest(:,23);
 [W_x, W_y, corr_list] = CCA(X', Y');
 X1=W_x'*X';
 Y1=W_y'*Y';