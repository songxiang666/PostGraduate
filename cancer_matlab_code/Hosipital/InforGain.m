function InforGain = gain(data) 
    [m, n] = size(data);
    InforGain = zeros(n-1,2);
    labels = data(:,n);
    for i=1:n
        tmp{i} = [];
        percen{i} = [];
        col = data(:,i);
        unicol = unique(col);
        %计算每一列有几类，并把每一类的信息熵和比例存储起来
        for j = 1:length(unicol)
            num = length(find(col==unicol(j)));
            pnum = length(find(col==unicol(j) & labels == 1));
            rate = pnum/num;
            if i==7
                rate = num/length(labels);
            end
            gain = -(rate*log2(rate)+(1-rate)*log2(1-rate));
            tmp{i}=[tmp{i} gain];
            percen{i}=[percen{i} num/length(col)];
        end
    end
    %整体信息熵
    InforEntropy = tmp{length(tmp)}(1);
    %将NAN转化为0
    for i = 1:length(tmp)
        tmp{i}(isnan(tmp{i})) = 0;
        %disp(tmp{i});
    end
    %求每一个属性列的信息增益
    for i = 1:length(percen)-1
        InforGain(i,:) = [i,roundn(InforEntropy-sum(tmp{i}.*percen{i}),-3)];
        disp(InforEntropy-sum(tmp{i}.*percen{i}));
    end
end

