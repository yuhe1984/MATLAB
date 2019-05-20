% ����˫��������
train_num=100;  % 0��1��������train_num���� ��������������Ŀ��
% 
train_i=(1: (105-1)/train_num: 105-(105-1)/train_num)';

%˫�������ݵ�Ĳ�������
alpha1=pi*(train_i-1)/25;
beta=0.4*((105-train_i)/(104)); %
x0=beta.*cos(alpha1);
y0=beta.*sin(alpha1);
z0=zeros(train_num,1);
x1=-beta.*cos(alpha1);
y1=-beta.*sin(alpha1);
z1=ones(train_num,1);

% �������˳��
k=rand(1,2*train_num);
[m,n]=sort(k);

train=[x0 y0 z0;x1,y1,z1]; 
train_label1=train(n(1:2*train_num),end)';    % 1*(2*train_num)
train_data1=train(n(1:2*train_num),1:end-1)'; % 2*(2*train_num)
      
% ��1ά��������2ά����� 0->[1 0], 1->[0 1]
for i=1:2*train_num
    switch train_label1(i)
        case 0
            train_label2(i,:)=[1 0];
        case 1
            train_label2(i,:)=[0 1];
    end
end

train_label=train_label2'; % 2*200
% ��ֵ��һ��
[train_data,train_datas]=mapminmax(train_data1);

% ��plot����ʾѵ������
plot(x0,y0,'r+');
hold on;
plot(x1,y1,'go');