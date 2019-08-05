function [LDCCA_train_data]=LDCCA2(X,Y,class_xy,options)
% switch n
%     case 1
%         v=[1 2 3];
%         X=Data(:,v)';
%         temp=Data;
%         temp(:,v)=[];%相当于Y=data-X,去掉X的部分
%         Y=temp';
%     case 2
%         v=[1 2 3 4 5 6 7 8 9 10];
%         X=Data(:,v)';
%         temp=Data;
%         temp(:,v)=[];%相当于Y=data-X,去掉X的部分
%         Y=temp';
%     case 16
%         v=[1 4 7 12 13 14];
%         X=Data(:,v)';
%         u=[15 16 18 19];
%         Y=Data(:,u)';
% end

B=[];
for i=1:size(class_xy,2);
    B=blkdiag(B,ones(class_xy(i),class_xy(i)));
end
CXY=X*B*Y';
CXX=X*X';
CYY=Y*Y';
RegSyy = 0;
RegSxx = 0;
if isfield(options, 'RegSxx')
    RegSxx = options.RegSxx;  
end

if isfield(options, 'RegSyy')
    RegSyy = options.RegSyy;
end
    CXX=CXX+ RegSxx*eye(size(CXX));
    CYY=CYY+ RegSyy*eye(size(CYY));
CCX=CXX^(1/2);
CCY=CYY^(1/2);
Z=inv(CCX)*CXY*inv(CCY);
[U,D,V]=svd(Z);
%choose [u1....ud]and [v1....vd],d<n
d=rank(D);% d: low dimensionality, or a specified value
U_bar=U(:,1:d);
V_bar=V(:,1:d);
%%----compute projection vectors
W_x=inv(CCX)*U_bar;
W_y=inv(CCY)*V_bar;
LDCCA_train_data=W_x'*X;
% LDCCA_train_data2=W_y'*Y;