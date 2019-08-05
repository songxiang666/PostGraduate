% function FE_ELM_NUM_main(n)
% switch n
% %     case 1
%         load yeast.mat
%         samp=yeast; 
%         Data=samp(:,2:9);
%         classnum=10;
%         per=[0,462,429,242,163,51,44,35,30,20,5];
%         dd=1;
%         perNum=per(2:size(per,2));
%         
%     case 2
%         load pendigits_train.mat
%         samp=pendigits_train;
%         Data=samp(:,2:17);
%         classnum=10;
%         per=[0,780,779,780,719,780,720,720,778,719,719];
%         dd=2;
%         perNum=per(2:size(per,2));
%     case 3
%         load letters_recognition.mat
%         samp=letters_recognition;
%         Data=samp(:,2:17);
%         classnum=26;
%         per=[0,789,766,736,805,768,775,773,734,755,747,739,761,792,783,753,803,783,758,748,796,813,764,752,787,786,734];
%         dd=3;
%         perNum=per(2:size(per,2));
%     case 4
%         load glass.mat
%         samp=glass;
%         Data=samp(:,1:10);
%         classnum=6;
%         per=[0,70,76,17,13,9,29];
%         dd=4;
%         perNum=per(2:size(per,2));
%     case 5
%         load newthyroid.mat
%         samp=newthyroid;
%         Data=samp(:,2:6);
%         classnum=3;
%         per=[0,150,35,30];
%         dd=5;
%         perNum=per(2:size(per,2));
%      case 6
%         load ionosphere.mat
%      case 7
%         load abalone.mat
%         samp=abalone;
%         Data=samp(:,2:9);
%         classnum=28;
%         per=[0,1,1,15,57,115,259,391,568,689,634,487,267,203,126,103,67,58,42,32,26,14,6,9,2,1,1,2,1];
%         dd=7;
%         perNum=per(2:size(per,2));
%     case 8
%         load Cardiotocographic.mat 
%         samp=Cardiotocographic;
%         Data=samp(:,2:22);
%         classnum=10;
%         per=[0,384,579,53,81,72,332,252,107,69,197];
%         dd=8;
%         perNum=per(2:size(per,2));
%     case 9
%         load segmentation.mat
%         samp=segmentation;
%         Data=samp(:,2:20);
%         classnum=7;
%         per=[0,30,30,30,30,30,30,30];
%         dd=9;
%         perNum=per(2:size(per,2));
%     case 10
%         load isolet.mat
%         samp=isolet;
%         Data=samp(:,1:617);
%         classnum=26;
%         per=[0,240,240,240,240,240,238,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240];
%         dd=10;
%         perNum=per(2:size(per,2));
%     case 11
%         load musk.mat
%         samp=musk;
%         Data=samp(:,1:166);
%         classnum=2;
%         per=[0,1017,5581];
%         dd=11;
%         perNum=per(2:size(per,2));
%     case 12
%         load spambase.mat
%         samp=spambase;
%         Data=samp(:,1:57);
%         classnum=2;
%         per=[0,1813,2788];
%         dd=12;
%         perNum=per(2:size(per,2));
%     case 13
%         load iris.mat
%         samp=iris;
%         Data=samp(:,2:5);
%         classnum=3;
%         per=[0,50,50,50];
%         dd=13;
%         perNum=per(2:size(per,2)); 
%
%       case 14
%         load wine.mat
%         samp=wine;
%         Data=samp(:,2:14);
%         classnum=3;
%         per=[0,59,71,48];
%         dd=14;
%         perNum=per(2:size(per,2)); 
%         
%         case 15
%          load glass.mat
%          samp=glass;
%         Data=samp(:,1:10);
%         classnum=6;
%         per=[0,70,76,17,13,9,29];
%         dd=15;
%         perNum=per(2:size(per,2)); 

%             case 16
%             load hearttest.mat
%             samp=hearttest;
%             Data=samp(:,1:22);
%             classnum=2;
%             per=[0,884,469];
%             dd=16;
%             perNum=per(2:size(per,2));
% end
load thir_non_excludeG_Cleaned.mat
samp= file_data;
Data=samp(:,2:5);
classnum=3;
per=[0,3847,1765,1600];
dd=17;
perNum=per(2:size(per,2));

%% 常数设置

%dim_PCA=5;
%dim_LDA=1;
%dim_LPP=40;
%dim_DLPP=40;
%dim_GLPP=40;

dim_CCA=3;
dim_semi_CCA=3;
dim_rank_CCA=3;
dim_LDCCA=3;
dim_LPCCA=3;
dim_DMPCCA=3;

lamma_DLPP=0.001;
beta_GLPP=1.2;

ActivationFunction='sig';
NumberofHiddenNeurons=20;

Data=NormalizeFea(Data);   
% %%  PCA输入设置
% fprintf('计算PCA数据')
% strat_time_PCA=cputime;
% options=[];
% [eigvector_PCA, eigvalue_PCA] = PCA(Data, options);
% end_time_PCA=cputime;
% time_PCA=end_time_PCA-strat_time_PCA;
% PCA_DATA=Data*eigvector_PCA(:,1:dim_PCA);
% %%  LDA输入设置
% fprintf('计算LDA数据')
% gnd=[];
% for i=1:classnum
%     gnd=[gnd;i*ones(perNum(i),1)];
% end
% strat_time_LDA=cputime;
% options = [];
% [eigvector_LDA, eigvalue_LDA] = LDA(gnd, options, Data);
% end_time_LDA=cputime;
% time_LDA=end_time_LDA-strat_time_LDA;
% LDA_DATA=Data*eigvector_LDA(:,1:dim_LDA);
% %% LPP输入设置
% strat_time_LPP=cputime;
% options = [];
% options.Metric = 'Euclidean';
% options.NeighborMode = 'KNN';
% options.k = 10;
% options.WeightMode = 'HeatKernel';
% options.t = 2;
% W = constructW(Data,options);
% [eigvector_LPP, eigvalue_LPP] = LPP(W, options, Data);
% end_time_LPP=cputime; 
% time_LPP=end_time_LPP-strat_time_LPP;
% LPP_DATA = Data*eigvector_LPP(:,1:dim_LPP);
% 
% %% DLPP输入设置
% strat_time_DLPP=cputime;
% [Zw,Zb,Zt]=DLPP(Data,classnum,per,10,2,2,2);
% [eigvector_DLPP, eigvalue_DLPP]=eig((Zb+lamma_DLPP*eye(size(Zb)))\Zw);
% [eigvalue_DLPP,index]=sort(diag(eigvalue_DLPP),'descend');
% eigvector_DLPP=eigvector_DLPP(:,index);
% end_time_DLPP=cputime;
% time_DLPP=end_time_DLPP-strat_time_DLPP;
% DLPP_DATA = Data*eigvector_DLPP(:,1:dim_DLPP);
% %% GLPP输入设置
% strat_time_GLPP=cputime;
% [Zw_glpp,Zb_glpp]=DLPP(Data,classnum,per,10,2,2,2);
% Delta=beta_GLPP*Zw+Zb;
% [eigvector_GLPP, eigvalue_GLPP]=eig(Delta);
% [eigvalue_GLPP,index]=sort(diag(eigvalue_GLPP),'descend');
% eigvector_GLPP=eigvector_GLPP(:,index);
% end_time_GLPP=cputime;
% time_GLPP=end_time_GLPP-strat_time_GLPP;
% GLPP_DATA = Data*eigvector_GLPP(:,1:dim_GLPP);

%% CCA输入设置
fprintf('计算CCA数据')
%strat_time_CCA=cputime;

v1 = [2 3];
v2 = [4 5];

% v1=[1 2];
% v2=[3 4];
X1=samp(:,v1)';X2=samp(:,v2)';
u=[1];
Y=samp(:,u)';
[train_data1] = CCA1(X1,Y);
[train_data2] = CCA1(X2,Y);

%end_time_CCA=cputime;
%time_CCA=end_time_CCA-strat_time_CCA;
CCA_DATA=[train_data1' train_data2'];

%% semi_CCA输入设置
fprintf('计算semi_CCA数据')
%strat_time_semi_CCA=cputime;
options = [];
options.RegSyy = 0;
options.RegSxx = 0;

v1 = [2 3];
v2 = [4 5];
% v1=[1 2];
% v2=[3 4];
X1=samp(:,v1)';X2=samp(:,v2)';
u=[1];
Y=samp(:,u)';
[semi_train_data1] = semi_cca1(X1,Y,samp(:,1),options);
[semi_train_data2] = semi_cca1(X2,Y,samp(:,1),options);

%end_time_semi_CCA=cputime;
%time_semi_CCA=end_time_semi_CCA-strat_time_semi_CCA;
% semi_train_data_a=semi_train_data1(1:dim_semi_CCA,:);
% semi_train_data_b=semi_train_data2(1:dim_semi_CCA,:);
semi_CCA_DATA=[semi_train_data1' semi_train_data2'];

%% rank_CCA输入设置
fprintf('计算rank_CCA数据')
%strat_time_rank_CCA=cputime;
options = [];
options.RegSyy = 0;
options.RegSxx = 0;

v1 = [2 3];
v2 = [4 5];
% v1=[1 2];
% v2=[3 4];
X1=samp(:,v1)';X2=samp(:,v2)';
u=[1];
Y=samp(:,u)';
[rank_train_data1] = rank_cca1(X1,Y,samp(:,1),options);
[rank_train_data2] = rank_cca1(X2,Y,samp(:,1),options);

%end_time_rank_CCA=cputime;
%time_rank_CCA=end_time_rank_CCA-strat_time_rank_CCA;
% rank_train_data_a=rank_train_data1(1:dim_rank_CCA,:);
% rank_train_data_b=rank_train_data2(1:dim_rank_CCA,:);
rank_CCA_DATA=[rank_train_data1' rank_train_data2'];

%% LDCCA输入设置
fprintf('计算LDCCA数据')
gnd=[];
for i=1:classnum
    gnd=[gnd;i*ones(perNum(i),1)];
end
options = [];
options.RegSyy = 0;
options.RegSxx = 0;
%strat_time_LDCCA=cputime;

v1 = [2 3];
v2 = [4 5];
% v1=[1 2];
% v2=[3 4];
X1=samp(:,v1)';X2=samp(:,v2)';
u=[1];
Y=samp(:,u)';
[LDCCA_train_data1] = LDCCA2(X1,Y,gnd,options);
[LDCCA_train_data2] = LDCCA2(X2,Y,gnd,options);

%end_time_LDCCA=cputime;
%time_LDCCA=end_time_LDCCA-strat_time_LDCCA;
% LDCCA_train_data_a=train_data1(1:dim_LDCCA,:);
% LDCCA_train_data_b=train_data2(1:dim_LDCCA,:);
LDCCA_DATA=[LDCCA_train_data1' LDCCA_train_data2'];

%% LPCCA输入设置
fprintf('计算LPCCA数据')
Neighbor_LP=6;
strat_time_LPCCA=cputime;

v1 = [2 3];
v2 = [4 5];
% v1=[1 2];
% v2=[3 4];
X1=samp(:,v1)';X2=samp(:,v2)';
u=[1];
Y=samp(:,u)';
[LPCCA_train_data1] = LPCCA1(X1,Y,Neighbor_LP);
[LPCCA_train_data2] = LPCCA1(X2,Y,Neighbor_LP);

%end_time_LPCCA=cputime;
%time_LPCCA=end_time_LPCCA-strat_time_LPCCA;
% LPCCA_train_data_a=train_data1(1:dim_LPCCA,:);
% LPCCA_train_data_b=train_data2(1:dim_LPCCA,:);
LPCCA_DATA=[LPCCA_train_data1' LPCCA_train_data2'];
%{
% DMPCCA输入设置
fprintf('计算DMPCCA数据')
gnd=[];
for i=1:classnum
    gnd=[gnd;i*ones(perNum(i),1)];
end
Neighbor_DMP=6;
strat_time_DMPCCA=cputime;

v1 = [2 3];
v2 = [4 5];
% v1=[1 2];
% v2=[3 4];
X1=samp(:,v1)';X2=samp(:,v2)';
u=[1];
Y=samp(:,u)';
[DMPCCA_train_data1] = DMPCCA(X1,Y,gnd,Neighbor_DMP);
[DMPCCA_train_data2] = DMPCCA(X2,Y,gnd,Neighbor_DMP);

%end_time_DMPCCA=cputime;
%time_DMPCCA=end_time_DMPCCA-strat_time_DMPCCA;
%% DMPCCA_train_data_a=train_data1(1:dim_DMPCCA,:);
%% DMPCCA_train_data_b=train_data2(1:dim_DMPCCA,:);
DMPCCA_DATA=[DMPCCA_train_data1' DMPCCA_train_data2'];
%}
%{
PCA_DATA=NormalizeFea(PCA_DATA);
LDA_DATA=NormalizeFea(LDA_DATA);
LPP_DATA=NormalizeFea(LPP_DATA);
DLPP_DATA=NormalizeFea(DLPP_DATA);
GLPP_DATA=NormalizeFea(GLPP_DATA);
CCA_DATA=NormalizeFea(CCA_DATA);
%}

%% 读取数据
%{
[trn_original,tst_original,trn_PCA,tst_PCA,trn_LDA,tst_LDA,trn_LPP,tst_LPP,trn_DLPP,tst_DLPP,trn_GLPP,tst_GLPP,trn_CCA,tst_CCA,trn_semi_CCA,tst_semi_CCA,trn_rank_CCA,tst_rank_CCA,trn_LDCCA,tst_LDCCA,trn_LPCCA,tst_LPCCA,trn_DMPCCA,tst_DMPCCA]=...
readFeaSam(Data',CCA_DATA',semi_CCA_DATA',rank_CCA_DATA',LDCCA_DATA',LPCCA_DATA',DMPCCA_DATA',dd);

Rate_sum_train=[];
Rate_sum_test=[];
Time_sum_train=[];
Time_sum_test=[];

time_fe=[];
for i=1:20
    rate_train=[];
    rate_test=[];
    time_train=[];
    time_test=[];
    



%% ELM
[TrainingTime_original, TestingTime_original, TrainingAccuracy_original, TestingAccuracy_original] = ELM(trn_original, tst_original, 1, NumberofHiddenNeurons, ActivationFunction);
% [TrainingTime_PCA, TestingTime_PCA, TrainingAccuracy_PCA, TestingAccuracy_PCA] = ELM(trn_PCA, tst_PCA, 1, NumberofHiddenNeurons, ActivationFunction);
% [TrainingTime_LDA, TestingTime_LDA, TrainingAccuracy_LDA, TestingAccuracy_LDA] = ELM(trn_LDA, tst_LDA, 1, NumberofHiddenNeurons, ActivationFunction);
% [TrainingTime_LPP, TestingTime_LPP, TrainingAccuracy_LPP, TestingAccuracy_LPP] = ELM(trn_LPP, tst_LPP, 1, NumberofHiddenNeurons, ActivationFunction);
% [TrainingTime_DLPP, TestingTime_DLPP, TrainingAccuracy_DLPP, TestingAccuracy_DLPP] = ELM(trn_DLPP, tst_DLPP, 1, NumberofHiddenNeurons, ActivationFunction);
% [TrainingTime_GLPP, TestingTime_GLPP, TrainingAccuracy_GLPP, TestingAccuracy_GLPP] = ELM(trn_GLPP, tst_GLPP, 1, NumberofHiddenNeurons, ActivationFunction);
[TrainingTime_CCA, TestingTime_CCA, TrainingAccuracy_CCA, TestingAccuracy_CCA] = ELM(trn_CCA, tst_CCA, 1, NumberofHiddenNeurons, ActivationFunction);
[TrainingTime_semi_CCA, TestingTime_semi_CCA, TrainingAccuracy_semi_CCA, TestingAccuracy_semi_CCA] = ELM(trn_semi_CCA, tst_semi_CCA, 1, NumberofHiddenNeurons, ActivationFunction);
[TrainingTime_rank_CCA, TestingTime_rank_CCA, TrainingAccuracy_rank_CCA, TestingAccuracy_rank_CCA] = ELM(trn_rank_CCA, tst_rank_CCA, 1, NumberofHiddenNeurons, ActivationFunction);
[TrainingTime_LDCCA, TestingTime_LDCCA, TrainingAccuracy_LDCCA, TestingAccuracy_LDCCA] = ELM(trn_LDCCA, tst_LDCCA, 1, NumberofHiddenNeurons, ActivationFunction);
[TrainingTime_LPCCA, TestingTime_LPCCA, TrainingAccuracy_LPCCA, TestingAccuracy_LPCCA] = ELM(trn_LPCCA, tst_LPCCA, 1, NumberofHiddenNeurons, ActivationFunction);
% [TrainingTime_DMPCCA, TestingTime_DMPCCA, TrainingAccuracy_DMPCCA, TestingAccuracy_DMPCCA] = ELM(trn_DMPCCA, tst_DMPCCA, 1, NumberofHiddenNeurons, ActivationFunction);



%% OS-ELM
% [TrainingTime_original, TestingTime_original, TrainingAccuracy_original, TestingAccuracy_original] = OSELM(trn_original, tst_original, 1, nHiddenNeurons, ActivationFunction, N0, Block);
% [TrainingTime_PCA, TestingTime_PCA, TrainingAccuracy_PCA, TestingAccuracy_PCA] = OSELM(trn_PCA, tst_PCA, 1, nHiddenNeurons, ActivationFunction, N0, Block);
% [TrainingTime_LDA, TestingTime_LDA, TrainingAccuracy_LDA, TestingAccuracy_LDA] = OSELM(trn_LDA, tst_LDA, 1, nHiddenNeurons, ActivationFunction, N0, Block);
% [TrainingTime_LPP, TestingTime_LPP, TrainingAccuracy_LPP, TestingAccuracy_LPP] = OSELM(trn_LPP, tst_LPP, 1, nHiddenNeurons, ActivationFunction, N0, Block);
% [TrainingTime_DLPP, TestingTime_DLPP, TrainingAccuracy_DLPP, TestingAccuracy_DLPP] = OSELM(trn_DLPP, tst_DLPP, 1, nHiddenNeurons, ActivationFunction, N0, Block);   
% [TrainingTime_GLPP, TestingTime_GLPP, TrainingAccuracy_GLPP, TestingAccuracy_GLPP] = OSELM(trn_GLPP, tst_GLPP, 1, nHiddenNeurons, ActivationFunction, N0, Block);


% rate_train=[TrainingAccuracy_original,TrainingAccuracy_PCA,TrainingAccuracy_LDA,TrainingAccuracy_LPP,TrainingAccuracy_DLPP,TrainingAccuracy_GLPP,TrainingAccuracy_CCA,TrainingAccuracy_semi_CCA,TrainingAccuracy_rank_CCA,TrainingAccuracy_LDCCA,TrainingAccuracy_LPCCA,TrainingAccuracy_DMPCCA];
% rate_test=[TestingAccuracy_original,TestingAccuracy_PCA,TestingAccuracy_LDA,TestingAccuracy_LPP,TestingAccuracy_DLPP,TestingAccuracy_GLPP,TestingAccuracy_CCA,TestingAccuracy_semi_CCA,TestingAccuracy_rank_CCA,TestingAccuracy_LDCCA,TestingAccuracy_LPCCA,TestingAccuracy_DMPCCA];
% time_train=[TrainingTime_original,TrainingTime_PCA,TrainingTime_LDA,TrainingTime_LPP,TrainingTime_DLPP,TrainingTime_GLPP,TrainingTime_CCA,TrainingTime_semi_CCA,TrainingTime_rank_CCA,TrainingTime_LDCCA,TrainingTime_LPCCA,TrainingTime_DMPCCA];
% time_test=[TestingTime_original,TestingTime_PCA,TestingTime_LDA,TestingTime_LPP,TestingTime_DLPP,TestingTime_GLPP,TestingTime_CCA,TestingTime_semi_CCA,TestingTime_rank_CCA,TestingTime_LDCCA,TestingTime_LPCCA,TestingTime_DMPCCA];

rate_train=[TrainingAccuracy_original,TrainingAccuracy_CCA,TrainingAccuracy_semi_CCA,TrainingAccuracy_rank_CCA,TrainingAccuracy_LDCCA,TrainingAccuracy_LPCCA,TrainingAccuracy_DMPCCA];
rate_test=[TestingAccuracy_original,TestingAccuracy_CCA,TestingAccuracy_semi_CCA,TestingAccuracy_rank_CCA,TestingAccuracy_LDCCA,TestingAccuracy_LPCCA,TestingAccuracy_DMPCCA];
time_train=[TrainingTime_original,TrainingTime_CCA,TrainingTime_semi_CCA,TrainingTime_rank_CCA,TrainingTime_LDCCA,TrainingTime_LPCCA,TrainingTime_DMPCCA];
time_test=[TestingTime_original,TestingTime_CCA,TestingTime_semi_CCA,TestingTime_rank_CCA,TestingTime_LDCCA,TestingTime_LPCCA,TestingTime_DMPCCA];

Rate_sum_train=[Rate_sum_train;rate_train];
Rate_sum_test=[Rate_sum_test;rate_test];
Time_sum_train=[Time_sum_train;time_train];
Time_sum_test=[Time_sum_test;time_test];


end
time_fe=[time_fe; time_CCA];
Rate_mean_train=mean(Rate_sum_train);
Rate_mean_test=mean(Rate_sum_test);

Rate_std_train=std(Rate_sum_train);
Rate_std_test=std(Rate_sum_test);

% RATE=[];
% for ww=1:20
%     ratess=[];
% 
% 
% ratess=[Percent,PercentD,PercentS,PercentX,PercentP];
% RATE=[RATE;ratess];
% end
% meanRate=mean(RATE)
% stdRate=std(RATE)

%}
