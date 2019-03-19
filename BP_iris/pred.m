function pred(x)
%init(2,5,3);%获得权重
% define the sigmoid function
f = @(x) 1/(1+exp(-x));
df = @(y) y .* (1 -y);


global prob;
prob.i_v = x;

%
total = prob.i_v * prob.ih_w;

for h = 1:prob.h_n
    prob.h_v(h) = f(total(h)-prob.h_t(h));
end

totall = prob.h_v * prob.ho_w;
for j = 1:prob.o_n
    prob.o_v(j) = f(totall(j)-prob.o_t(j));
end
%prob.o_v = o_v;

% e_k = [];
% y_delta2 = 0;
% for j = 1:o_n
%     y_delta2 = y_delta2 + (o_v(j)-y(j)) * (o_v(j) - y(j));
% end
% e_k(end+1) = (y_delta2/2);