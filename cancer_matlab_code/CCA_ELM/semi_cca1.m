function [semi_train_data]=semi_cca1(X,Y,label,options)
% semi-supervised canonical correlation analysis
% X,Y : training set, X is p*n matrix，Y is q*n matrix
% label: the label of Cconstrains(rateing positive constraint 
%        to negative constraint equal 1 or not equal 1(in 
%        other word , their proportion is not the same))
% d: low dimensionality
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
%%-------centering data-----
[m_X,n_X]=size(X);
bar_x=mean(X,2);
CX=X-repmat(bar_x,1,n_X);
[m_Y,n_Y]=size(Y);
bar_y=mean(Y,2);
CY=Y-repmat(bar_y,1,n_Y);
%%------compute constraint matrices----
M=zeros(n_X,n_X);
C=zeros(n_X,n_X);
%set the corresponding position in constraint matrices to be '1'according
% to the original position i and j of cx(i)and cx(j);
% label=1,2 stand for positive constraint and negative constraint,respectively
for i=1:n_X
    for j=1:n_X
        if label(i)==label(j)
            M(i,j)=1;
            M(j,i)=1;
        else
            C(i,j)=1;
            C(j,i)=1;
        end
    end
end        
%%-----compute covariance matrices----
S=eye(n_X,n_X)+M+C;
CXX=CX*CX';
CYY=CY*CY';
CXY_bar=CX*S*CY';
%%---compute matrix Z----
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
Z=inv(CCX)*CXY_bar*inv(CCY);
[U,D,V]=svd(Z);
%choose [u1....ud]and [v1....vd],d<n
d=rank(D);% d: low dimensionality, or a specified value
U_bar=U(:,1:d);
V_bar=V(:,1:d);
%%----compute projection vectors
W_x=inv(CCX)*U_bar;
W_y=inv(CCY)*V_bar;
% W_x=abs(W_x);
% W_y=abs(W_y);
semi_train_data=W_x'*X;
% semi_train_data2=W_y'*Y;