%LDCCA
function [W_x,W_y,CW,CB]=LDCCA(X,Y,classnum,allnum)
[nSmpx,nFeax] = size(X);
if issparse(X)
   X = full(X);
end
sampleMeanx = mean(X);
X = (X - repmat(sampleMeanx,nSmpx,1));
[nSmpy,nFeay] = size(Y);
if issparse(Y)
   Y = full(Y);
end
sampleMeany = mean(Y);
Y = (Y - repmat(sampleMeany,nSmpy,1));
%% computing local within-class covariance and local between-class covariance
CW=zeros(nFeax,nFeax);
CB=zeros(nFeax,nFeax);
%寻找近邻
sitew=[];
for i=1:classnum
    XX=X(:,100*(i-1)+1:100*i);
    trainlabel=100*(i-1)+1:1:100*i;
    trage=[];
    for j=1:allnum
        BB=XX;
        DD=trainlabel;
        BB(:,j)=[];
        DD(:,j)=[];
        result=knn(BB',DD',X(:,j)',5);
        trage=[trage;result];
    end
    sitew=[sitew,trage];%每1列为一组
end
siteb=[];
for i=1:classnum
    xx=X(:,100*(i-1)+1:100*i);
    for j=1:classnum
        if i~=j
            yy=X(:,100*(j-1)+1:100*j);
            tral=100*(j-1)+1:1:100*j;
            resut=knn(yy',tral',xx',5);
            siteb=[siteb,resut];%每4列为一组
        end
    end
end     
%计算cw,cb
for i=1:classnum
    site=sitew(:,i);
    for j=1:allnum
        CW(j,site(j))=1;
        CW(site(j),j)=1;
    end
end
for i=1:classnum
    sit=siteb(:,4*(i-1)+1:4*i);
    for j=1:classnum-1
        ss=sit(:,j);
        for k=1:allnum
            CB(k,ss(k))=1;
            CB(ss(k),k)=1;
        end
    end
end
gamma=0.1;
CXY=X*(CW-gamma*CB)*Y';
CXX=X*X';
CYY=Y*Y';
CCX=CXX^(1/2);
CCY=CYY^(1/2);
Z=inv(CCX)*CXY*inv(CCY);
[U,D,V]=svd(Z);
%choose [u1....ud]and [v1....vd],d<n
d=rank(D);% d: low dimensionality, or a specified value
U_bar=U(:,1:d);
V_bar=V(:,1:d);
%%----compute projection vectors
W_x=inv(CCX)*U_bar;
W_y=inv(CCY)*V_bar;
end
% function nind=knearestn(XX)
% % Fingding local within-class nn 
% xx=size(XX,2);
% dist=zeros(xx,xx);
% for i=1:xx
%     for j=1:xx
%         dist(i,j)=norm(XX(:,i)-XX(:,j));
%     end
% end
% nind=[];
% for i=1:xx
%     [ss,site]=sort(dist(i,:),'ascend');
%     nind=[nind;site(2)];
% end
% end
% function nindb=knearestnb(XX)
% % Fingding local between-class nn
% X1=XX(:,101:500);
% XX1=XX(:,1:100);
% X2=[XX(:,1:100),XX(:,201:500)];
% XX2=XX(:,101:200);
% X3=[XX(:,1:200),XX(:,301:500)];
% XX3=XX(:,201:300);
% X4=[XX(:,1:300),XX(:,401:500)];
% XX4=XX(:,301:400);
% X5=XX(:,1:400);
% XX5=XX(:,401:500);
% xx1=size(XX1,2);
% dist1=zeros(xx1,500-xx1);
% for i=1:xx1
%     for j=1:500-xx1
%         dist1(i,j)=norm(XX1(:,i)-X1(:,j));
%     end
% end
% nind1=[];
% for i=1:xx1
%     [ss1,site1]=sort(dist1(i,:),'ascend');
%     nind1=[nind1;site1(2)];
% end
% xx2=size(XX2,2);
% dist2=zeros(xx2,500-xx2);
% for i=1:xx2
%     for j=1:500-xx2
%         dist2(i,j)=norm(XX2(:,i)-X2(:,j));
%     end
% end
% nind2=[];
% for i=1:xx2
%     [ss2,site2]=sort(dist2(i,:),'ascend');
%     nind2=[nind2;site2(2)];
% end
% xx3=size(XX3,2);
% dist3=zeros(xx3,500-xx3);
% for i=1:xx3
%     for j=1:500-xx3
%         dist3(i,j)=norm(XX3(:,i)-X3(:,j));
%     end
% end
% nind3=[];
% for i=1:xx3
%     [ss3,site3]=sort(dist3(i,:),'ascend');
%     nind3=[nind3;site3(2)];
% end
% 
% xx4=size(XX4,2);
% dist4=zeros(xx4,500-xx4);
% for i=1:xx4
%     for j=1:500-xx4
%         dist4(i,j)=norm(XX4(:,i)-X4(:,j));
%     end
% end
% nind4=[];
% for i=1:xx4
%     [ss4,site4]=sort(dist4(i,:),'ascend');
%     nind4=[nind4;site4(2)];
% end
% 
% xx5=size(XX5,2);
% dist5=zeros(xx5,500-xx5);
% for i=1:xx5
%     for j=1:500-xx5
%         dist5(i,j)=norm(XX5(:,i)-X5(:,j));
%     end
% end
% nind5=[];
% for i=1:xx5
%     [ss5,site5]=sort(dist5(i,:),'ascend');
%     nind5=[nind5;site5(2)];
% end
% nindb=[nind1;nind2;nind3;nind4;nind5];
% end




