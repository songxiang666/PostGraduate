% eigvector ��������
% eigvalue  ����ֵ
% elapse    ʱ������

function [eigvector, eigvalue, elapse] = PCA(data, options)
%% PCA	Principal Component Analysis
%
%	Usage:
%       [eigvector, eigvalue] = PCA(data, options)
%       [eigvector, eigvalue] = PCA(data)
% 
%             Input:
%               data       - Data matrix. Each row vector of fea is a data point.
%
%     options.ReducedDim   - The dimensionality of the reduced subspace. If 0,
%                         all the dimensions will be kept. 
%                         Default is 0. 
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           data point (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%               eigvalue  - The sorted eigvalue of PCA eigen-problem. 
%
%	Examples:
% 			fea = rand(7,10);
% 			[eigvector,eigvalue] = PCA(fea,4);
%           Y = fea*eigvector;
% 
%   version 2.2 --Feb/2009 
%   version 2.1 --June/2007 
%   version 2.0 --May/2007 
%   version 1.1 --Feb/2006 
%   version 1.0 --April/2004 
%
%   Written by Deng Cai (dengcai2 AT cs.uiuc.edu)
%                                                   

%%  �ж��Ƿ���ѡ���ά����
if (~exist('options','var'))
   options = [];
end

%% ������ά��
ReducedDim =0 ;
if isfield(options,'ReducedDim')
    ReducedDim = options.ReducedDim;
end
%% �жϽ�ά���Ƿ����0<ReducedDim<����ά��
[nSmp,nFea] = size(data);
if (ReducedDim > nFea) || (ReducedDim <=0)
    ReducedDim = nFea; %�������ά������С�ڵ���0���͵�������ά��
end

%%  ��ʱ��������tic/toc��׼ȷ
tmp_T = cputime;

%% ���ϡ�����Ĵ洢��ʽ��sparse storage organization���򷵻��߼�1�����򷵻��߼�0
if issparse(data)
    data = full(data); %��ϡ�����תΪȫ����
end
%% ����ȥ��ֵ��
sampleMean = mean(data,1); %������ÿһ�о�ֵ 
data = (data - repmat(sampleMean,nSmp,1)); %ȥ��ֵ
 

%% 
if nFea/nSmp > 1.0713 % ά����������
    % This is an efficient method which computes the eigvectors of
	% of A*A^T (instead of A^T*A) first, and then convert them back to
	% the eigenvectors of A^T*A.    
    ddata = data*data';
    ddata = max(ddata, ddata'); %��ȡÿһ��Ԫ�����

    dimMatrix = size(ddata,2); %ddata��ά��
    if dimMatrix > 1000 && ReducedDim < dimMatrix/10  % using eigs to speed up!
        option = struct('disp',0);
        %d = eigs(A,k,sigma,opts)  
        %����k���������ֵ,
        %sigmaȡֵ��'lm' ��ʾ����ֵ��������ֵ��'sm' ����ֵ��С����ֵ��
        %��ʵ�Գ����⣺'la'��ʾ�������ֵ��'sa'Ϊ��С����ֵ��
        %�ԷǶԳƺ͸������⣺'lr' ��ʾ���ʵ����'sr' ��ʾ��Сʵ����
        %'li' ��ʾ����鲿��'si'��ʾ��С�鲿
       [eigvector, eigvalue] = eigs(ddata,ReducedDim,'la',option);%���������� ����ֵ
        eigvalue = diag(eigvalue);%�����ԽǾ���
    else
        [eigvector, eigvalue] = eig(ddata); %ֱ����������������ֵ
        eigvalue = diag(eigvalue); %�Խ���

        [junk, index] = sort(-eigvalue); %junk����õ�����index������-Ϊ�˴Ӵ�С
        eigvalue = eigvalue(index);
        eigvector = eigvector(:, index); %��������������
    end

    clear ddata;
    
    maxEigValue = max(abs(eigvalue)); %abs����ֵ�����������ֵ
    eigIdx = find(abs(eigvalue)/maxEigValue < 1e-12); %��ȡ��С������ֵ
    eigvalue (eigIdx) = []; %����Ӧ������ֵ�ÿ�
    eigvector (:,eigIdx) = [];%����Ӧ�����������ÿ�

    % ����
    eigvector = data'*eigvector;		% Eigenvectors of A^T*A
	eigvector = eigvector*diag(1./(sum(eigvector.^2).^0.5)); % Normalization
else
    ddata = data'*data;
    ddata = max(ddata, ddata');

    dimMatrix = size(ddata,2);
    if dimMatrix > 1000 & ReducedDim < dimMatrix/10  % using eigs to speed up!
        option = struct('disp',0);
        [eigvector, eigvalue] = eigs(ddata,ReducedDim,'la',option);
        eigvalue = diag(eigvalue);
    else
        [eigvector, eigvalue] = eig(ddata);
        eigvalue = diag(eigvalue);

        [junk, index] = sort(-eigvalue);
        eigvalue = eigvalue(index);
        eigvector = eigvector(:, index);
    end
    
    clear ddata;
    maxEigValue = max(abs(eigvalue));
    eigIdx = find(abs(eigvalue)/maxEigValue < 1e-12);
    eigvalue (eigIdx) = [];
    eigvector (:,eigIdx) = [];
end

%% 
if ReducedDim < length(eigvalue)
    eigvalue = eigvalue(1:ReducedDim);
    eigvector = eigvector(:, 1:ReducedDim);
end

%% 
if isfield(options,'PCARatio') %PCA��
    sumEig = sum(eigvalue);
    sumEig = sumEig*options.PCARatio; %����ֵ��X����
    sumNow = 0;
    for idx = 1:length(eigvalue)
        sumNow = sumNow + eigvalue(idx);
        if sumNow >= sumEig
            break;
        end
    end
    eigvector = eigvector(:,1:idx);
end
%% ���㻨�ѵ�ʱ��
elapse = cputime - tmp_T;
