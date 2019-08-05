function [W_X,W_Y,corr]=DLPCCA_ED(X,Y,B,SXX,SYY,options)
% DLPCCA_ED
%% πÊ‘ÚªØ
if (~exist('options','var'))
   options = [];
end
% PrjX = 1;
% PrjY = 1;
% if isfield(options, 'PrjX')
%     PrjX = options.PrjX;
% end
% if isfield(options, 'PrjY')
%     PrjY = options.PrjY;
% end

if isfield(options, 'RegSxx')
    RegSxx = options.RegSxx;  
end

if isfield(options, 'RegSyy')
    RegSyy = options.RegSyy;
end
    SXX=SXX+ RegSxx*eye(size(SXX));
    SYY=SYY+ RegSyy*eye(size(SYY));
    disp('DX,DY...'); 
    DX=X*SXX^(1/2);
    DY=Y*SYY^(1/2);

    % Compute the SVD of X
    [X_U, X_Sigma, X_V] = svd(DX, 'econ');
    X_rank = rank(X_Sigma);
    X_U = X_U(:, 1:X_rank);
    X_Sigma =X_Sigma(1:X_rank, 1:X_rank);
    X_V =X_V(:, 1:X_rank);
    % Compute the SVD of Y
    [Y_U, Y_Sigma, Y_V] = svd(DY, 'econ');
    rank_Y = rank(Y_Sigma);
    Y_U = Y_U(:, 1:rank_Y);
    Y_Sigma = Y_Sigma(1:rank_Y, 1:rank_Y);
    Y_V = Y_V(:, 1:rank_Y);
    % Compute the projection vectors
    HY=B*SXX^(-1/2);
    MY=inv(Y_Sigma)*Y_U'*Y*HY;
    HX=B*SYY^(-1/2);
    MX=inv(X_Sigma)*X_U'*X*HX;

    [P_X,MX_Sigma,Q_X]=svd(MX, 'econ');
    rank_MX = rank(MX);
    P_X = P_X(:, 1:rank_MX);
    MX_sigma = diag(MX_Sigma*MX_Sigma');
    MX_sigma = MX_sigma(1:rank_MX);
    corr=sqrt(MX_sigma);
    W_X = X_U * inv(X_Sigma) * P_X;
    
    [P_Y,MY_Sigma,Q_Y]=svd(MY, 'econ');
    rank_MY = rank(MY);
    P_Y = P_Y(:, 1:rank_MY);
    MY_sigma = diag(MY_Sigma*MY_Sigma');
    MY_sigma = MY_sigma(1:rank_MY);
    corr=sqrt(MY_sigma);
    W_Y = Y_U * inv(Y_Sigma) * P_Y;
    ranK=min(rank_MX,rank_MY);
    W_Y=W_Y(:,1:ranK);
    W_X=W_X(:,1:ranK);

    
    