function e_k = errorr(y)
global prob;
% for j = 1:prob.o_n
%     a = (prob.o_v(j)-y(j)) * (prob.o_v(j) - y(j));
%     y_delta2 = y_delta2 + a;
% end

a = (prob.o_v-y) .* (prob.o_v - y);
y_delta2 = sum(a);
e_k = y_delta2/2;