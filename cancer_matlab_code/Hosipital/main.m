load CancerData2.mat
samp=CancerData2(:,2:41);
Data=samp;
classnum=2;
per=[0,2658,2385];
dd=1;
perNum=per(2:size(per,2));

ActivationFunction='sig';
NumberofHiddenNeurons=50;

Data1=NormalizeFea(Data);

% options.ReducedDim=40;
% [eigvectorr, eigvaluer, elapser] = PCA(Data1, options);
% Data = Data1*eigvectorr;


% ELM分类
[trn_original,tst_original]=readFeaSam(Data1',dd);

Rate_sum_train=[];
Rate_sum_test=[];
Time_sum_train=[];
Time_sum_test=[];
for i=1:30
    rate_train=[];
    rate_test=[];
    time_train=[];
    time_test=[];
    %% ELM
    [TrainingTime_original, TestingTime_original, TrainingAccuracy_original, TestingAccuracy_original] = ELM(trn_original, tst_original, 1, NumberofHiddenNeurons, ActivationFunction);

    rate_train=[TrainingAccuracy_original];
    rate_test=[TestingAccuracy_original];
    time_train=[TrainingTime_original];
    time_test=[TestingTime_original];

    Rate_sum_train=[Rate_sum_train;rate_train];
    Rate_sum_test=[Rate_sum_test;rate_test];
    Time_sum_train=[Time_sum_train;time_train];
    Time_sum_test=[Time_sum_test;time_test];
    disp(['迭代次数：',num2str(i)]) ;
end
Rate_mean_train=mean(Rate_sum_train);
Rate_mean_test=mean(Rate_sum_test);

Rate_time_train=mean(Time_sum_train);
Rate_time_test=mean(Time_sum_test);

Rate_std_train=std(Rate_sum_train);
Rate_std_test=std(Rate_sum_test);
