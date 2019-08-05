function [rank_train_data]=rank_cca1(X,Y,label,options)
% ranking canonical correlation analysis
% X,Y : training set, X is p*n matrix，Y is q*n matrix
% label: the label of constrains(level relevance degree label)
% d: low dimensionality
%  注意 X，Y 包含了没有标志的数据 ，这时label我们设成0


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
%%------- alpha gamma :weighting parameters
alpha=0.4;
gamma=0.8;
%%-------centering data-----
[m_X,n_X]=size(X);
bar_x=mean(X,2);
CX=X-repmat(bar_x,1,n_X);
[m_Y,n_Y]=size(Y);
bar_y=mean(Y,2);
CY=Y-repmat(bar_y,1,n_Y);
CX=X;
CY=Y;
%%------compute constraint matrices----
CAA=zeros(n_X,n_X);
CBB=zeros(n_X,n_X);
CCC=zeros(n_X,n_X);
CAB=zeros(n_X,n_X);
CAC=zeros(n_X,n_X);
CBC=zeros(n_X,n_X);
%set the corresponding position in constraint matrices to be '1'according
% to the original position i and j of cx(i)and cx(j);
% level relevance degree label=1,2,3 stand for very relevance，relevance and irrelevance,respectively
for i=1:n_X
    for j=1:n_Y
    if label(i)==1&&label(j)==1
        CAA(i,j)=1;CAA(j,i)=1;
    else if label(i)==2&&label(j)==2
             CBB(i,j)=1;CBB(j,i)=1;
        else if label(i)==1&&label(j)==2
                 CAB(i,j)=1;CAB(j,i)=1;
            else if label(i)==1&&label(j)==3
                     CAC(i,j)=1;CAC(j,i)=1;
                else if label(i)==2&&label(j)==3
                         CBC(i,j)=1;CBC(j,i)=1;
                    else if label(i)==3&&label(j)==3
                             CCC(i,j)=1;CCC(j,i)=1;
                        end
                    end
                end
            end
        end
    end
    end
end 

%%-----compute covariance matrices----
%S=eye(n_X,n_X)+gamma.*CAA+gamma.*CBB+gamma*alpha.*CAB+(1-gamma).*CCC+(1-gamma).*CAC+(1-gamma).*CBC;
S=eye(n_X,n_X)+gamma.*CAA+gamma.*CBB+gamma*alpha.*CAB-(1-gamma).*CCC-(1-gamma).*CAC-(1-gamma).*CBC;
CXX=CX*CX';
CYY=CY*CY';
CXY_bar=CX*S*CY';
% CXX=CXX+0.01*eye(size(CXX));%解决非奇异问题，lmmda选择有待探究，一般来说越小越好
% CYY=CYY+0.01*eye(size(CYY));
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
corr=diag(D);
corr=corr(1:d);
rank_train_data=W_x'*X;
% rank_train_data2=W_y'*Y;

