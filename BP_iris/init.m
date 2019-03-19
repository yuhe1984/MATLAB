function init(ni,nh,no)%ni为第一层神经元数量，nh为第二层，no为第三层，lr为学习率
%初始化
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
