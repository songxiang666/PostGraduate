function [DMPCCA_train_data]=DMPCCA(X,Y,class_xy,Neighbor)
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
%% 输入数据库
% d1 = 10;
% d2 = 100;
% n = 200;
% X = rand(d1, n);
% Y = rand(d2, n);
% X=[191 189 193 162 189 182 211;
%     36 37 38 35 35 36 38;
%     50 52 58 62 46 56 56];
% Y=[5 2 12 12 13 4 8;
%     162 110 101 105 155 101 101];
% class_xy=[1,2,4];

% load cereal
% Fat;Fiber;Potass;Protein;Sodium;Sugars;Vitamins;
% X=[Fat,Fiber,Potass,Protein,Sodium,Sugars,Vitamins]';
% Weight;Carbo;Calories;
% Y=[Weight,Carbo,Calories]';
% class_xy=[20,1,22,1,13,1,19];
%% GAUSS数据
% mu1=[10.18,0.66]';
% sigma1=[22 3.75; 3.75 4];
% X1=mvnrnd(mu1,sigma1,80);
% mu2=[5,-5]';
% sigma2=[1 0; 0 1];
% X2=mvnrnd(mu2,sigma2,100);
% X=[X1;X2]';
% mu=[1,1]';
% sigma=[0.01 0; 0 0.01];
% E=mvnrnd(mu,sigma,180);
% W=[0.6 -sqrt(1/2);0.8 sqrt(1/2)];
% Y=W'*X+E';
% subplot(1,3,1)
% %figure(1)
% plot(X(1,1:80),Y(1,1:80),'r*')
% hold on
% plot(X(1,81:180)',Y(1,81:180),'r.')
% hold on
% plot(X(2,1:80)',Y(2,1:80),'+')
% hold on
% plot(X(2,81:180)',Y(2,81:180),'o')
% class_xy=[80,100];
% %% car shuju
% [X,Y,class_xy]=car;
% %% IRIS
% load iris.mat
% X=iris(:,2:3)';
% Y=iris(:,4:5)';
% class_xy=[50,50,50];
%% 正式程序
% [nSmpx,nFeax] = size(X);
% if issparse(X)
%    data = full(X);
% end
% sampleMeanx = mean(X);
% X = (X - repmat(sampleMeanx,nSmpx,1));
% [nSmpy,nFeay] = size(Y);
% if issparse(Y)
%    Y = full(Y);
% end
% sampleMeany = mean(Y);
% Y = (Y - repmat(sampleMeany,nSmpy,1));
% figure(7)
% plot(X(1,:),Y(1,:),'r.')
% hold on
% plot(X(2,:)',Y(2,:),'+')


%% 输入类信息,计算判别矩阵B
disp('Computing discrimination matrix...'); 
%class_xy=ones(1,nFeax);%需要输入
B=[];
for i=1:size(class_xy,2);
    B=blkdiag(B,ones(class_xy(i),class_xy(i)));
end
%B=eye(n,n);这时的DLPCCA和LPCCA只差分子
%% 计算相似矩阵
disp('Computing similiarity matrix...'); 
options = [];
options.NeighborMode = 'KNN';
options.k = Neighbor;%可以调整
options.WeightMode = 'HeatKernel';
options.t = 1;%可以调整
S_X = constructW(X',options);
S_Y = constructW(Y',options);  

S_X=full(S_X);%+ones(size(S_X));
S_X=full(S_X);%+ones(size(S_X));
SX=S_X.*S_X;
SY=S_Y.*S_Y;
D_XX=diag(full(sum(SX,2)));%+ones(size(diag(full(sum(SX,2)))));
D_YY=diag(full(sum(SY,2)));%+ones(size(diag(full(sum(SY,2)))));
% rank(D_XX)
% rank(D_YY)
S_XX=D_XX-SX;
S_YY=D_YY-SY;
%% 求解 DLPCCA
disp('Solving DLPCCA...'); 
options.RegSxx=1;
options.RegSyy=1;
[W_x,W_y,corr]=DLPCCA_ED(X,Y,B,S_XX,S_YY,options);
DMPCCA_train_data=W_x'*X;
% DMPCCA_train_data2=W_y'*Y;

% S_XY=X*B*Y';
% RegSyy = 0;
% RegSxx = 0;
% if isfield(options, 'RegSxx')
%     RegSxx = options.RegSxx;  
% end
% 
% if isfield(options, 'RegSyy')
%     RegSyy = options.RegSyy;
% end
%     S_XX=S_XX+ RegSxx*eye(size(S_XX));
%     S_YY=S_YY+ RegSyy*eye(size(S_YY));
% S_XX=S_XX^(1/2);
% S_YY=S_YY^(1/2);
% S_XX=X*S_XX*X';
% S_YY=Y*S_YY*Y';
% Z=inv(S_XX)*S_XY*inv(S_YY);
% [U,D,V]=svd(Z);
% %choose [u1....ud]and [v1....vd],d<n
% d=rank(D);% d: low dimensionality, or a specified value
% U_bar=U(:,1:d);
% V_bar=V(:,1:d);
% %%----compute projection vectors
% W_X=inv(S_XX)*U_bar;
% W_Y=inv(S_YY)*V_bar;










% P_X=W_X'*X;
% P_Y=W_Y'*Y;
% for i=1:dims
%     %figure(i+1)
%     subplot(1,2,2)
%     %subplot(dims,dims,i)
% %     plot(P_X(1,1:80),P_Y(1,1:80),'r.')
% %     hold on
% %     plot(P_X(1,81:180),P_Y(1,81:180),'+')
% %     figure(dims+i)
% %     plot(P_X(2,1:80),P_Y(2,1:80),'r.')
% %     hold on
% %     plot(P_X(2,81:180),P_Y(2,81:180),'+')
% plot(P_X(1,1:50),P_Y(1,1:50),'.')
% hold on
% plot(P_X(1,51:100),P_Y(1,51:100),'ro')
% hold on
% plot(P_X(1,101:150),P_Y(1,101:150),'s')
% end
% L_X = constructW(P_X',options);
% L_Y = constructW(P_Y',options); 
% E_X=norm(full(S_X-L_X))
% E_Y=norm(full(S_Y-L_Y))
% %% 执行CCA
% options.PrjX = 1;
% options.PrjY = 1;
% options.RegX = 0.1;
% options.RegY = 1;
% [W_x, W_y, corr_list] = CCA(X, Y, options);
% P_X1 = W_x' * X;
% P_Y1= W_y' * Y;
% %figure(dims+2)
% subplot(1,2,1)
% %     plot(X_p(1,1:80),Y_p(1,1:80),'r.')
% %     hold on
% %     plot(X_p(1,81:180),Y_p(1,81:180),'+')
% %plot(X_p(1,:),Y_p(1,:),'.')
% plot(P_X1(1,1:50),P_Y1(1,1:50),'.')
% hold on
% plot(P_X1(1,51:100),P_Y1(1,51:100),'ro')
% hold on
% plot(P_X1(1,101:150),P_Y1(1,101:150),'s')
% L_X1 = constructW(P_X1',options);
% L_Y1 = constructW(P_Y1',options); 
% E_X1=norm(full(S_X-L_X1))
% E_Y1=norm(full(S_Y-L_Y1))










% RegX = 0;
% RegS = 0;
% if isfield(options, 'RegX')
%     RegX = options.RegX;
% end
% if isfield(options, 'RegS')
%     RegS = options.RegS;
% end
% if isfield(options, 'no_dims')
%     no_dims = options.no_dims;
% end
% if ~exist('eig_impl', 'var') 
%    eig_impl = 'Matlab'; 
% end  
% if rank(S_XX)==min(size(S_XX))
%     H=B/inv(S_XX)*B;
% else
%     H=B/inv(S_XX+RegS*eye(size(S_XX)))*B;
% end
%  DP =Y* S_YY*Y'; 
%  LP =Y*H*Y'; 
% % DP =Y*Y'; 
% % LP =Y*X'/inv(X*X'+0.0001*eye(size(X*X')))*X*Y'; 
% %     DP = (DP + DP') / 2; 
% %     LP = (LP + LP') / 2; 
% % Perform eigenanalysis of generalized eigenproblem (as in LEM) 
% if size(Y, 2) > 200 && no_dims < (size(Y, 2) / 2) 
%    if strcmp(eig_impl, 'JDQR') 
%       options.Disp = 0; 
%       options.LSolver = 'bicgstab'; 
%       [W_Y, corr] = jdqz(LP, DP, no_dims, 'SA', options); 
%    else 
%       options.disp = 0; 
%       options.issym = 1; 
%       options.isreal = 1; 
%       [W_Y, corr] = eigs(LP, DP, no_dims, 'SA', options); 
%    end 
% else 
%    [W_Y, corr] = eig(LP, DP); 
% end 
% % Sort eigenvalues in descending order and get largest eigenvectors 
% [corr, ind] = sort(diag(corr), 'descend'); 
% W_Y = W_Y(:,ind(1:no_dims)); 
% % eigIdx = find(corr < 1e-3);
% % corr (eigIdx) = [];
% % W_Y(:,eigIdx) = [];
% corr=sqrt(corr);
% W_X=[];
% for i=1:no_dims
% W_x=1/corr(i).*inv(X*X'+0.0001*eye(size(X*X')))*X*Y'*W_Y(:,i);
% W_X=[W_X,W_x];
% end
% % for i=1:no_dims
% %     if rank(S_XX)==min(size(S_XX))
% %         if rank(X*X')==min(size(X*X'))
% %             alpha_X=1/corr(i)*X/inv(X'*X)/inv(S_XX)*B*Y'*W_Y(:,i);
% %             W_X=[W_X,alpha_X];
% %         else
% %             alpha_X=1/corr(i)*X/inv(X'*X+RegX*eye(size(X'*X)))/inv(S_XX)*B*Y'*W_Y(:,i);
% %             W_X=[W_X,alpha_X];
% %         end
% %     else
% %         if rank(X*X')==min(size(X*X'))
% %             alpha_X=1/corr(i)*X/inv(X'*X)/inv(S_XX+RegS*eye(size(S_XX)))*B*Y'*W_Y(:,i);
% %             W_X=[W_X,alpha_X];
% %         else
% %             alpha_X=1/corr(i)*X/inv(X'*X+RegX*eye(size(X'*X)))/inv(S_XX+RegS*eye(size(S_XX)))*B*Y'*W_Y(:,i);
% %             W_X=[W_X,alpha_X];
% %         end
% %     end
% % end
% %% 约简数据库，投影数据
% disp('Reducing dababase...');
% R_X=W_X'*X;
% R_Y=W_Y'*Y;
% figure(1)
% plot(R_X,R_Y)
% % subplot(2,2,2)
% % %plot(X(1,:),Y(1,:),'.')
% % plot(X(1,1:80),Y(1,1:80),'r*')
% % hold on
% % plot(X(1,81:180)',Y(1,81:180),'r.')
% % hold on
% % plot(X(2,1:80)',Y(2,1:80),'*')
% % hold on
% % plot(X(2,81:180)',Y(2,81:180),'.')
% % subplot(2,2,3)
% % plot(R_X(1,:),R_Y(1,:),'.')
% subplot(2,2,3)
% % plot(X(2,:),Y(2,:),'.')
% 
% % plot(R_X(2,:),R_Y(2,:),'.')
% %figure(2)
% plot(R_X(1,1:80),R_Y(1,1:80),'r*')
% hold on
% plot(R_X(1,81:180)',R_Y(1,81:180),'.')
% % figure(3)
% % plot(R_X(2,1:50)',R_Y(2,1:50),'r*')
% % hold on
% % plot(R_X(2,51:150)',R_Y(2,51:150),'.')
% % 
%  subplot(2,2,4)

 
 
 
 
 
 
 
%% 求解DLPCCA，利用对偶求解
%  W_X=[];
%  W_Y=[];
%  lamm=0.0001;
% %d=    %约简后的维数
% if det(S_XX)~=0
%     H=Y*B*S_XX^(-1)*B*Y';
%     if det(Y*S_YY*Y')~=0
%         [W_Y,corr]=eig(H,Y*S_YY*Y');
%         corr=diag(corr);
%         corr=sqrt(real(corr));
%     else
%         [W_Y,corr]=eig(H,Y*S_YY*Y'+lamm*eye(size(Y*S_YY*Y')));
%         corr=diag(corr);
%         corr=sqrt(real(corr));
%     end
%     for i=1:size(corr,1)
%         if det(X*X')~=0
%             alpha_X=1/corr(i)*X*(X'*X)^(-1)*S_XX^(-1)*B*Y'*W_Y(:,i);
%             W_X=[W_X,alpha_X];
%         else
%             alpha_X=1/corr(i)*X*(X'*X+lamm*eye(size(X'*X)))^(-1)*S_XX^(-1)*B*Y'*W_Y(:,i);
%             W_X=[W_X,alpha_X];
%         end
%     end
% else
%     lamm=0.0001;
%     S_XX=S_XX+lamm*eye(size(S_XX));
%     H=Y*B*S_XX^(-1)*B*Y';
%     if det(Y*S_YY*Y')~=0
%         [W_Y,corr]=eig(H,Y*S_YY*Y');
%         corr=diag(corr);
%         corr=sqrt(real(corr));
%     else
%         [W_Y,corr]=eig(H,Y*S_YY*Y');
%         corr=diag(corr);
%         corr=sqrt(real(corr));
%     end
%     for i=1:size(corr,1)
%         if det(X*X')~=0
%             alpha_X=1/corr(i)*X*(X'*X)^(-1)*S_XX^(-1)*B*Y'*W_Y(:,i);
%             W_X=[W_X,alpha_X];
%         else
%             alpha_X=1/corr(i)*X*(X'*X+lamm*eye(size(X'*X)))^(-1)*S_XX^(-1)*B*Y'*W_Y(:,i);
%             W_X=[W_X,alpha_X];
%         end
%     end
% end


