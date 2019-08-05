%LPCCA
function [LPCCA_train_data]=LPCCA1(X,Y,Neighbor)

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

options = [];
options.NeighborMode = 'KNN';
options.k = Neighbor;%可以调整
options.WeightMode = 'HeatKernel';
options.t = 1;%可以调整
S_X = constructW(X',options);
S_Y = constructW(Y',options); 
S_X=full(S_X);
S_Y=full(S_Y);

SX=S_X.*S_X;
SY=S_Y.*S_Y;
SXY=S_X.*S_Y;
D_XX=diag(full(sum(SX,2)));
D_YY=diag(full(sum(SY,2)));
D_XY=diag(full(sum(SXY,2)));
S_XX=D_XX-SX;
S_YY=D_YY-SY;
S_XY=D_XY-SXY;
options.RegSxx=1;
options.RegSyy=1;
[W_x,W_y,corr]=DLPCCA_ED(X,Y,S_XY,S_XX,S_YY,options);
LPCCA_train_data=W_x'*X;
% LPCCA_train_data2=W_y'*Y;