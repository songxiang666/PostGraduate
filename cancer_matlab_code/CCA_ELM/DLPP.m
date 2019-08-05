function [Zw,Zb,Zt]=DLPP(Data,classnum,per,a1,a2,b1,b2)
%注意per第一个数字必须是零，每一类中的个数
%classnum 总的类别数
%a1,a2,b1,b2都是邻域参数
lo=size(Data,1);
M=[];
fprintf('计算DLPP M')
    options=[];
    options.NeighborMode = 'KNN';
    options.k = a1;
    options.WeightMode = 'HeatKernel';
    options.t = a2;
    M = constructW(Data,options);

F=[];
for i=2:classnum+1
    x_aver=Data(per(i-1)+1:per(i-1)+per(i),:);
    f=mean(x_aver,1);
    F=[F; f]
end
    options=[];
    options.NeighborMode = 'KNN';
    options.k = b1;
    options.WeightMode = 'HeatKernel';
    options.t = b2;
    B = constructW(F,options);
fprintf('计算DLPP B')
% for i=1:size(F,2)
%     for j=1:size(F,2)
%         B(i,j)=exp(-(norm(F(:,i)-F(:,j))^2)/b2);
%     end
% end


L=zeros(lo,lo);
for i=1:lo
    L(i,i)=sum(M(i,:));
end
E=zeros(classnum,classnum);
for i=1:size(B,1)
    E(i,i)=sum(B(i,:));
end
Zw=Data'*(L-M)*Data;
Zb=F'*(E-B)*F;
Zt=Zb+Zw;
fprintf('计算DLPP 结束')