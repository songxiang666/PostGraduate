function [trn_original,tst_original]=...
    readFeaSam(Data,n)

%每一行为一个数据
switch n
    case 1
        %noG,mat
        pos=[1, 2658, 2659, 5043];
        trnnum=[1772, 1590];
        allnum=[2658,2385];
        classnum=2;
  
end
%把每一类存放成cell形式
sam_ORI={};
% sam_PCA={};sam_LDA={};sam_LPP={};sam_DLPP={};sam_GLPP={};
for i=1:classnum
    sam_ORI{i}=Data(:,pos(2*(i-1)+1):pos(2*i));
%     sam_PCA{i}=PCA_DATA(:,pos(2*(i-1)+1):pos(2*i));
%     sam_LDA{i}=LDA_DATA(:,pos(2*(i-1)+1):pos(2*i));
%     sam_LPP{i}=LPP_DATA(:,pos(2*(i-1)+1):pos(2*i));
%     sam_DLPP{i}=DLPP_DATA(:,pos(2*(i-1)+1):pos(2*i));
%     sam_GLPP{i}=GLPP_DATA(:,pos(2*(i-1)+1):pos(2*i));
end
trn_original=[];tst_original=[];
% trn_PCA=[];tst_PCA=[];trn_LDA=[];tst_LDA=[];
% trn_LPP=[];tst_LPP=[];trn_DLPP=[];tst_DLPP=[];trn_GLPP=[];tst_GLPP=[];

trn_original_Y=[];tst_original_Y=[];
% trn_PCA_Y=[];tst_PCA_Y=[];trn_LDA_Y=[];tst_LDA_Y=[];
% trn_LPP_Y=[];tst_LPP_Y=[];trn_DLPP_Y=[];tst_DLPP_Y=[];trn_GLPP_Y=[];tst_GLPP_Y=[];
for i=1:classnum
    aa=sam_ORI{i};
%     bb=sam_PCA{i};
%     cc=sam_LDA{i};
%     dd=sam_LPP{i};
%     ee=sam_DLPP{i};
%     ff=sam_GLPP{i};
    randsamp=randperm(allnum(i));
    randsamp=randsamp(1:trnnum(i));
    trn_original=[trn_original,aa(:,sort(randsamp))];
%     trn_PCA=[trn_PCA,bb(:,sort(randsamp))];
%     trn_LDA=[trn_LDA,cc(:,sort(randsamp))];
%     trn_LPP=[trn_LPP,dd(:,sort(randsamp))];
%     trn_DLPP=[trn_DLPP,ee(:,sort(randsamp))];
%     trn_GLPP=[trn_GLPP,ff(:,sort(randsamp))];
    aa(:,sort(randsamp))=[];
%     bb(:,sort(randsamp))=[];
%     cc(:,sort(randsamp))=[];
%     dd(:,sort(randsamp))=[];
%     ee(:,sort(randsamp))=[];
%     ff(:,sort(randsamp))=[];
    tst_original=[tst_original,aa];
%     tst_PCA=[tst_PCA,bb];
%     tst_LDA=[tst_LDA,cc];
%     tst_LPP=[tst_LPP,dd];
%     tst_DLPP=[tst_DLPP,ee];
%     tst_GLPP=[tst_GLPP,ff];
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
end

trn_original=[trn_original_Y',trn_original'];
% trn_PCA=[trn_PCA_Y',trn_PCA'];
% trn_LDA=[trn_LDA_Y',trn_LDA'];
% trn_LPP=[trn_LPP_Y',trn_LPP'];
% trn_DLPP=[trn_DLPP_Y',trn_DLPP'];
% trn_GLPP=[trn_GLPP_Y',trn_GLPP'];


tst_original=[tst_original_Y',tst_original'];
% tst_PCA=[tst_PCA_Y',tst_PCA'];
% tst_LDA=[tst_LDA_Y',tst_LDA'];
% tst_LPP=[tst_LPP_Y',tst_LPP'];
% tst_DLPP=[tst_DLPP_Y',tst_DLPP'];
% tst_GLPP=[tst_GLPP_Y',tst_GLPP'];

end

