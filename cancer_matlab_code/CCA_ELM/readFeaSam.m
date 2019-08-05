% function [trn_original,tst_original,trn_PCA,tst_PCA,trn_LDA,tst_LDA,trn_LPP,tst_LPP,trn_DLPP,tst_DLPP,trn_GLPP,tst_GLPP,trn_CCA,tst_CCA,trn_semi_CCA,tst_semi_CCA,trn_rank_CCA,tst_rank_CCA,trn_LDCCA,tst_LDCCA,trn_LPCCA,tst_LPCCA,trn_DMPCCA,tst_DMPCCA]=...
%     readFeaSam(Data,PCA_DATA,LDA_DATA,LPP_DATA,DLPP_DATA,GLPP_DATA,CCA_DATA,semi_CCA_DATA,rank_CCA_DATA,LDCCA_DATA,LPCCA_DATA,DMPCCA_DATA,n)
function [trn_original,tst_original,trn_PCA,tst_PCA,trn_LDA,tst_LDA,trn_LPP,tst_LPP,trn_DLPP,tst_DLPP,trn_GLPP,tst_GLPP,trn_CCA,tst_CCA,trn_semi_CCA,tst_semi_CCA,trn_rank_CCA,tst_rank_CCA,trn_LDCCA,tst_LDCCA,trn_LPCCA,tst_LPCCA,trn_DMPCCA,tst_DMPCCA]=...
    readFeaSam(Data,CCA_DATA,semi_CCA_DATA,rank_CCA_DATA,LDCCA_DATA,LPCCA_DATA,DMPCCA_DATA,n)

%ÿһ��Ϊһ������
switch n
    case 1
        %  load yeast.mat
        
        pos=[1,462,463,891,892,1133,1134,1296,1297,1347,1348,1391,1392,1426,1427,1456,1457,1476,1477,1481];
        trnnum=[231,214,121,81,25,22,17,15,10,3];
        allnum=[462,429,242,163,51,44,35,30,20,5];
        classnum=10;
    case 2
        %   load pendigits_train.mat
        
        pos=[1,780,781,1559,1560,2339,2340,3058,3059,3838, 3839,4558,4559,5278,5279,6056,6057,6775,6776,7494];
        trnnum=[390,390,390,360,390,360,360,389,360,360];
        allnum=[780,779,780,719,780,720,720,778,719,719];
        classnum=10;
    case 3
        %   load letters_recognition.mat
        
        pos=[1,789,790,1555,1556,2291,2292,3096,3097,3864,3865,4639,4640,5412,5413,...
            6146,6147,6901,6902,7648,7649,8387,8388,9148,9149,9940,9941,10723,10724,...
            11476,11477,12279,12280,13062,13063,13820,13821,14568,14569,15364,15365,...
            16177,16178,16941,16942,17693,17694,18480,18481,19266,19267,20000];
        trnnum=[395,383,368,403,384,388,387,367,378,374,370,381,396,392,377,402,392,379,374,398,407,382,376,394,393,367];
        allnum=[789,766,736,805,768,775,773,734,755,747,739,761,792,783,753,803,783,758,748,796,813,764,752,787,786,734];
        classnum=26;
    case 4
        %   load glass.mat
        
        pos=[1,70,71,146,147,163,164,176,177,185,186,214];
        trnnum=[30,30,8,5,4,15];
        allnum=[70,76,17,13,9,29];
        classnum=6;
    case 5
        %   load newthyroid.mat
        
        pos=[1,150,151,185,186,215];
        trnnum=[75,18,15];
        allnum=[150,35,30];
        classnum=3;
    case 6
       
    case 7
        %         load abalone.mat
        pos=[1,1,2,2,3,17,18,74,75,189,190,448,449,839,840,1407,1408,2096,2097,2730,2731,3217,3218,3484,3485,3687,3688,3813,3814,3916,3917,3983,3984,4041,4042,4083,4084,4115,4116,4141,4142,4155,4156,4161,4162,4170,4171,4172,4173,4173,4174,4174,4175,4176,4177,4177];
        trnnum=[1,1,8,29,58,130,196,284,345,317,244,134,102,63,52,34,29,21,16,13,7,3,5,1,1,1,1,1];
        allnum=[1,1,15,57,115,259,391,568,689,634,487,267,203,126,103,67,58,42,32,26,14,6,9,2,1,1,2,1];
        classnum=28;
    case 8
        %         load Cardiotocographic.mat
        pos=[1,384,385,963,964,1016,1017,1097,1098,1169,1170,1501,1502,1753,1754,1860,1861,1929,1930,2126];
        trnnum=[192,290,27,41,36,166,126,54,35,99];
        allnum=[384,579,53,81,72,332,252,107,69,197];
        classnum=10;
    case 9
        %         load segmentation.mat
        pos=[1,30,31,60,61,90,91,120,121,150,151,180,181,210];
        trnnum=[15,15,15,15,15,15,15];
        allnum=[30,30,30,30,30,30,30];
        classnum=7;
    case 10
        %         load isolet.mat
        pos=[1,240,241,480,481,720,721,960,961,1200,1201,1438,1439,1678,1679,...
            1918,1919,2158,2159,2398,2399,2638,2639,2878,2879,3118,3119,3358,3359,3598,3599,...
            3838,3839,4078,4079,4318,4319,4558,4559,4798,4799,5038,5039,5278,5279,5518,5519,5758,5759,5998,5999,6238];
        trnnum=[120,120,120,120,120,119,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120,120];
        allnum=[240,240,240,240,240,238,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240];
        classnum=26;
    case 11
        %         load musk.mat
        pos=[1,1017,1018,6598];
        trnnum=[509,2791];
        allnum=[1017,5581];
        classnum=2;
    case 12
        %         load spambase.mat
        pos=[1,1813,1814,4601];
        trnnum=[907,1394];
        allnum=[1813,2788];
        classnum=2;
    case 13
        %         load iris.mat
        pos=[1,50,51,100,101,150];
        trnnum=[25,25,25];
        allnum=[50,50,50];
        classnum=3;
    case 14
        %         load wine.mat
        pos=[1,59,60,130,131,178];
        trnnum=[30,36,24];
        allnum=[59,71,48];
        classnum=3;
    case 15
         %         load glass.mat
        pos=[1,70,71,146,147,163,164,176,177,185,186,214];
        trnnum=[35,38,9,7,5,15];
        allnum=[70,76,17,13,9,29];
        classnum=6;
    case 16
        %         load hearttest.mat
        pos=[1,884,885,1353];
        trnnum=[442,235];
        allnum=[884,469];
        classnum=2;
     case 17
        %noG,mat
        pos=[1,3847,3848,7212];
        trnnum=[2539,2240];
        allnum=[3847,3365];
        classnum=2;
end
%��ÿһ���ų�cell��ʽ
sam_ORI={};sam_PCA={};sam_LDA={};sam_LPP={};sam_DLPP={};sam_GLPP={};
sam_CCA={};sam_semi_CCA={};sam_rank_CCA={};sam_LDCCA={};sam_LPCCA={};
sam_DMPCCA={};
for i=1:classnum
    sam_ORI{i}=Data(:,pos(2*(i-1)+1):pos(2*i));
%     sam_PCA{i}=PCA_DATA(:,pos(2*(i-1)+1):pos(2*i));
%     sam_LDA{i}=LDA_DATA(:,pos(2*(i-1)+1):pos(2*i));
%     sam_LPP{i}=LPP_DATA(:,pos(2*(i-1)+1):pos(2*i));
%     sam_DLPP{i}=DLPP_DATA(:,pos(2*(i-1)+1):pos(2*i));
%     sam_GLPP{i}=GLPP_DATA(:,pos(2*(i-1)+1):pos(2*i));
    sam_CCA{i}=CCA_DATA(:,pos(2*(i-1)+1):pos(2*i));
    sam_semi_CCA{i}=semi_CCA_DATA(:,pos(2*(i-1)+1):pos(2*i));
    sam_rank_CCA{i}=rank_CCA_DATA(:,pos(2*(i-1)+1):pos(2*i));    
    sam_LDCCA{i}=LDCCA_DATA(:,pos(2*(i-1)+1):pos(2*i));
    sam_LPCCA{i}=LPCCA_DATA(:,pos(2*(i-1)+1):pos(2*i));
    sam_DMPCCA{i}=DMPCCA_DATA(:,pos(2*(i-1)+1):pos(2*i));
end
trn_original=[];tst_original=[];trn_PCA=[];tst_PCA=[];trn_LDA=[];tst_LDA=[];
trn_LPP=[];tst_LPP=[];trn_DLPP=[];tst_DLPP=[];trn_GLPP=[];tst_GLPP=[];
trn_CCA=[];tst_CCA=[];trn_semi_CCA=[];tst_semi_CCA=[];trn_rank_CCA=[];tst_rank_CCA=[];
trn_LDCCA=[];tst_LDCCA=[];trn_LPCCA=[];tst_LPCCA=[];trn_DMPCCA=[];tst_DMPCCA=[];

trn_original_Y=[];tst_original_Y=[];trn_PCA_Y=[];tst_PCA_Y=[];trn_LDA_Y=[];tst_LDA_Y=[];
trn_LPP_Y=[];tst_LPP_Y=[];trn_DLPP_Y=[];tst_DLPP_Y=[];trn_GLPP_Y=[];tst_GLPP_Y=[];
trn_CCA_Y=[];tst_CCA_Y=[];trn_semi_CCA_Y=[];tst_semi_CCA_Y=[];trn_rank_CCA_Y=[];tst_rank_CCA_Y=[];
trn_LDCCA_Y=[];tst_LDCCA_Y=[];trn_LPCCA_Y=[];tst_LPCCA_Y=[];trn_DMPCCA_Y=[];tst_DMPCCA_Y=[];
for i=1:classnum
    aa=sam_ORI{i};
%     bb=sam_PCA{i};
%     cc=sam_LDA{i};
%     dd=sam_LPP{i};
%     ee=sam_DLPP{i};
%     ff=sam_GLPP{i};
    gg=sam_CCA{i};
    hh=sam_semi_CCA{i};
    ii=sam_rank_CCA{i};   
    jj=sam_LDCCA{i};
    kk=sam_LPCCA{i};
    ll=sam_DMPCCA{i};
    randsamp=randperm(allnum(i));
    randsamp=randsamp(1:trnnum(i));
    trn_original=[trn_original,aa(:,sort(randsamp))];
%     trn_PCA=[trn_PCA,bb(:,sort(randsamp))];
%     trn_LDA=[trn_LDA,cc(:,sort(randsamp))];
%     trn_LPP=[trn_LPP,dd(:,sort(randsamp))];
%     trn_DLPP=[trn_DLPP,ee(:,sort(randsamp))];
%     trn_GLPP=[trn_GLPP,ff(:,sort(randsamp))];
    trn_CCA=[trn_CCA,gg(:,sort(randsamp))];
    trn_semi_CCA=[trn_semi_CCA,hh(:,sort(randsamp))];
    trn_rank_CCA=[trn_rank_CCA,ii(:,sort(randsamp))];   
    trn_LDCCA=[trn_LDCCA,jj(:,sort(randsamp))];
    trn_LPCCA=[trn_LPCCA,kk(:,sort(randsamp))];
    trn_DMPCCA=[trn_DMPCCA,ll(:,sort(randsamp))];
    aa(:,sort(randsamp))=[];
%     bb(:,sort(randsamp))=[];
%     cc(:,sort(randsamp))=[];
%     dd(:,sort(randsamp))=[];
%     ee(:,sort(randsamp))=[];
%     ff(:,sort(randsamp))=[];
    gg(:,sort(randsamp))=[];
    hh(:,sort(randsamp))=[];    
    ii(:,sort(randsamp))=[];  
    jj(:,sort(randsamp))=[];  
    kk(:,sort(randsamp))=[]; 
    ll(:,sort(randsamp))=[]; 
    tst_original=[tst_original,aa];
%     tst_PCA=[tst_PCA,bb];
%     tst_LDA=[tst_LDA,cc];
%     tst_LPP=[tst_LPP,dd];
%     tst_DLPP=[tst_DLPP,ee];
%     tst_GLPP=[tst_GLPP,ff];
    tst_CCA=[tst_CCA,gg];
    tst_semi_CCA=[tst_semi_CCA,hh];
    tst_rank_CCA=[tst_rank_CCA,ii];    
    tst_LDCCA=[tst_LDCCA,jj];    
    tst_LPCCA=[tst_LPCCA,kk];    
    tst_DMPCCA=[tst_DMPCCA,ll]; 
end

for i=1:classnum
    trn_original_Y=[trn_original_Y,i*ones(1,trnnum(i))];
    tst_original_Y=[tst_original_Y,i*ones(1,allnum(i)-trnnum(i))];
    
%     trn_PCA_Y=[trn_PCA_Y,i*ones(1,trnnum(i))];
%     tst_PCA_Y=[tst_PCA_Y,i*ones(1,allnum(i)-trnnum(i))];
%     
%     trn_LDA_Y=[trn_LDA_Y,i*ones(1,trnnum(i))];
%     tst_LDA_Y=[tst_LDA_Y,i*ones(1,allnum(i)-trnnum(i))];
%     
%     trn_LPP_Y=[trn_LPP_Y,i*ones(1,trnnum(i))];
%     tst_LPP_Y=[tst_LPP_Y,i*ones(1,allnum(i)-trnnum(i))];
%     
%     trn_DLPP_Y=[trn_DLPP_Y,i*ones(1,trnnum(i))];
%     tst_DLPP_Y=[tst_DLPP_Y,i*ones(1,allnum(i)-trnnum(i))];
%     
%     trn_GLPP_Y=[trn_GLPP_Y,i*ones(1,trnnum(i))];
%     tst_GLPP_Y=[tst_GLPP_Y,i*ones(1,allnum(i)-trnnum(i))];
    
    trn_CCA_Y=[trn_CCA_Y,i*ones(1,trnnum(i))];
    tst_CCA_Y=[tst_CCA_Y,i*ones(1,allnum(i)-trnnum(i))];
    
    trn_semi_CCA_Y=[trn_semi_CCA_Y,i*ones(1,trnnum(i))];
    tst_semi_CCA_Y=[tst_semi_CCA_Y,i*ones(1,allnum(i)-trnnum(i))];
 
    trn_rank_CCA_Y=[trn_rank_CCA_Y,i*ones(1,trnnum(i))];
    tst_rank_CCA_Y=[tst_rank_CCA_Y,i*ones(1,allnum(i)-trnnum(i))];
 
    trn_LDCCA_Y=[trn_LDCCA_Y,i*ones(1,trnnum(i))];
    tst_LDCCA_Y=[tst_LDCCA_Y,i*ones(1,allnum(i)-trnnum(i))];
    
    trn_LPCCA_Y=[trn_LPCCA_Y,i*ones(1,trnnum(i))];
    tst_LPCCA_Y=[tst_LPCCA_Y,i*ones(1,allnum(i)-trnnum(i))];
    
    trn_DMPCCA_Y=[trn_DMPCCA_Y,i*ones(1,trnnum(i))];
    tst_DMPCCA_Y=[tst_DMPCCA_Y,i*ones(1,allnum(i)-trnnum(i))];    
end

trn_original=[trn_original_Y',trn_original'];
% trn_PCA=[trn_PCA_Y',trn_PCA'];
% trn_LDA=[trn_LDA_Y',trn_LDA'];
% trn_LPP=[trn_LPP_Y',trn_LPP'];
% trn_DLPP=[trn_DLPP_Y',trn_DLPP'];
% trn_GLPP=[trn_GLPP_Y',trn_GLPP'];
trn_CCA=[trn_CCA_Y',trn_CCA'];
trn_semi_CCA=[trn_semi_CCA_Y',trn_semi_CCA'];
trn_rank_CCA=[trn_rank_CCA_Y',trn_rank_CCA'];
trn_LDCCA=[trn_LDCCA_Y',trn_LDCCA'];
trn_LPCCA=[trn_LPCCA_Y',trn_LPCCA'];
trn_DMPCCA=[trn_DMPCCA_Y',trn_DMPCCA'];

tst_original=[tst_original_Y',tst_original'];
% tst_PCA=[tst_PCA_Y',tst_PCA'];
% tst_LDA=[tst_LDA_Y',tst_LDA'];
% tst_LPP=[tst_LPP_Y',tst_LPP'];
% tst_DLPP=[tst_DLPP_Y',tst_DLPP'];
% tst_GLPP=[tst_GLPP_Y',tst_GLPP'];
tst_CCA=[tst_CCA_Y',tst_CCA'];
tst_semi_CCA=[tst_semi_CCA_Y',tst_semi_CCA'];
tst_rank_CCA=[tst_rank_CCA_Y',tst_rank_CCA'];
tst_LDCCA=[tst_LDCCA_Y',tst_LDCCA'];
tst_LPCCA=[tst_LPCCA_Y',tst_LPCCA'];
tst_DMPCCA=[tst_DMPCCA_Y',tst_DMPCCA'];
end
