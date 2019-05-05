function standardRPROP()
global prob;
x = prob.inputData;
y = prob.outputData;
batch = length(prob.inputData);
e = [];
for i = 1:prob.runIndexFinish
e_k = [];
    for i = 1:batch
        pred(x(i,:));
        e_kk = errorr(y(i,:));
        e_k = horzcat(e_k,e_kk);
        bptrain(y(i,:));
    end
    ee = sum(e_k) / length(e_k);
    e = horzcat(e,ee);
end
%[mine,index] = min(e)
figure;
plot(e);
title('¹Ì¶¨Ñ§Ï°ÂÊ')
end