% 产生双螺旋数据
train_num=100;  % 0和1的样本各train_num个。 可以设置任意数目。
% 
train_i=(1: (105-1)/train_num: 105-(105-1)/train_num)';

%双螺旋数据点的产生方程
alpha1=pi*(train_i-1)/25;
beta=0.4*((105-train_i)/(104)); %
x0=beta.*cos(alpha1);
y0=beta.*sin(alpha1);
z0=zeros(train_num,1);
x1=-beta.*cos(alpha1);
y1=-beta.*sin(alpha1);
z1=ones(train_num,1);

% 随机打乱顺序
k=rand(1,2*train_num);
[m,n]=sort(k);

train=[x0 y0 z0;x1,y1,z1]; 
train_label1=train(n(1:2*train_num),end)';    % 1*(2*train_num)
train_data1=train(n(1:2*train_num),1:end-1)'; % 2*(2*train_num)
      
% 把1维的输出变成2维的输出 0->[1 0], 1->[0 1]
for i=1:2*train_num
    switch train_label1(i)
        case 0
            train_label2(i,:)=[1 0];
        case 1
            train_label2(i,:)=[0 1];
    end
end

train_label=train_label2'; % 2*200
% 均值归一化
[train_data,train_datas]=mapminmax(train_data1);

% 在plot上显示训练样本
plot(x0,y0,'r+');
hold on;
plot(x1,y1,'go');