%读入数据
tic
clc;clear;
global prob;
rand('seed',1);
x = zeros(150,4);
y = zeros(150,3);
n = 1000;
batch = length(y);
%[f1,f2,f3,f4,class] = textread('trainData.txt' , '%f%f%f%f%f',150);
[f1,f2,f3,f4,class] = textread('totalData.txt' , '%f%f%f%f%f',150);
[c1,c2,c3] = textread('class.txt' , '%f%f%f',150);
x(:,1) = f1;
x(:,2) = f2;
x(:,3) = f3;
x(:,4) = f4;
y(:,1) = c1;
y(:,2) = c2;
y(:,3) = c3;

%固定学习率
init(4,5,3);
lr = 0.1;
e = [];
for i = 1:n%用变量代替
    e_k = [];
    for i = 1:batch
        pred(x(i,:));
        e_kk = errorr(y(i,:));
        %e_k(end+1) = e_kk
        e_k = horzcat(e_k,e_kk);
        ee = sum(e_k) / length(e_k);
        bptrain(y(i,:),lr);
        %e(end+1) = e;
    end
    e = horzcat(e,ee);
end

figure;
plot(e);
title('固定学习率');
% for i = 1:150
%     pred(x(i,:));
%     prob.o_v
% end

%动态学习率
init(4,5,3);                                      
lr = 0.1;
ep = [];
lr1 = [];
e1 = [];
j = 0;
for i = 1:n
    e_kp = [];
    j = j + 1;
    for i = 1:batch
        pred(x(i,:));
        e_kkp = errorr(y(i,:));
        e_kp = horzcat(e_kp,e_kkp);
        eep = sum(e_kp) / length(e_kp);
        bptrain(y(i,:),lr);
    end
    if j == 1
        e1 = e_kp;
    end
    if j > 1
        for h = 1:length(e1)
            if e1(h) > e_kp(h)
                e1(h) = e_kp(h);
                lr1 = horzcat(lr1,lr);
            end
        end
        if rand(1,1) < 0.01
            lr = rand(1,1);
         else
            a = mean(lr1);
            b = std(lr1);
            lr = normrnd(a,b);
        end
    end
    ep = horzcat(ep,eep);
end

figure;
plot(ep);
title('动态学习率');
% for i = 1:150
%     pred(x(i,:));
%     prob.o_v
% end
toc