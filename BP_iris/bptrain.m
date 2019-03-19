function bptrain(y,lr)
global prob;
f = @(x) 1/(1+exp(-x));
df = @(y) y .* (1 -y);

o_grid = zeros(1,prob.o_n);
% for j = 1:prob.o_n
%     o_grid(j) = (y(j) - prob.o_v(j)) * df(prob.o_v(j));
% end
o_grid = (y - prob.o_v) .* df(prob.o_v);

h_grid = zeros(1,prob.h_n);
% for h = 1:prob.h_n
%     for j = 1:prob.o_n
%         a = prob.ho_w(h,j) * o_grid(j);
%         h_grid(h) = h_grid(h) + a;
%     end
%     h_grid(h) = h_grid(h) * df(prob.h_v(h));
% end
a =  o_grid .* prob.ho_w;
% h_grid = (a(:,1) + a(:,2) + a(:,3))';
h_grid = sum(a,2)';
h_grid = h_grid .* df(prob.h_v);

%¸üÐÂ
% for h = 1:prob.h_n
%     for j = 1:prob.o_n
%         prob.ho_w(h,j) = prob.ho_w(h,j) + lr * o_grid(j) * prob.h_v(h);
%     end
% end
b = o_grid' .* prob.h_v;
prob.ho_w = prob.ho_w + lr * b';

% for i = 1:prob.i_n
%     for h =1:prob.h_n
%         prob.ih_w(i,h) = prob.ih_w(i,h) + lr * h_grid(h) * prob.i_v(i);
%     end
% end
c = h_grid' .* prob.i_v;
prob.ih_w = prob.ih_w + lr * c';

% for j = 1:prob.o_n
%     prob.o_t(j) = prob.o_t(j) - lr * o_grid(j);
% end
prob.o_t = prob.o_t - lr * o_grid;

% for h = 1:prob.h_n
%     prob.h_t(h) = prob.h_t(h) - lr * h_grid(h);
% end
prob.h_t = prob.h_t - lr * h_grid; 