function init(ni,nh,no)%niΪ��һ����Ԫ������nhΪ�ڶ��㣬noΪ�����㣬lrΪѧϰ��
%��ʼ��
global prob;
prob.i_n = ni;
prob.h_n = nh;
prob.o_n = no;

prob.i_v = zeros(1,prob.i_n);
prob.h_v = zeros(1,prob.h_n);
prob.o_v = zeros(1,prob.o_n);

prob.ih_w = rand(prob.i_n,prob.h_n);
prob.ho_w = rand(prob.h_n,prob.o_n);

prob.h_t = rand(1,prob.h_n);
prob.o_t = rand(1,prob.o_n);

% define the sigmoid function
f = @(x) 1/(1+exp(-x));
df = @(y) y * (1 -y);
