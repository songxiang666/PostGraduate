%LPCCA
function [W_X,W_Y,corr,S_X,S_Y]=LPCCA(X,Y,Neighbor)
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
[W_X,W_Y,corr]=DLPCCA_ED(X,Y,S_XY,S_XX,S_YY,options);