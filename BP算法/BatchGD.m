function BatchGD()
tic
global prob alg;
x = prob.inputData;
y = prob.outputData;
%x = mapminmax(x);
lr = alg.learnRate;
ek = [];
for i = 1:prob.runIndexFinish
%     err = Batchpred(x,y,lr);
%     eb = horzcat(eb,err);
     alg.outputAtLayers{1} = x;
     total = alg.outputAtLayers{1} * alg.w{1} - alg.t{1};
     alg.outputAtLayers{2} = logsig(total);
     totall = alg.outputAtLayers{2} * alg.w{2} - alg.t{2};
     alg.outputAtLayers{3} = logsig(totall);
     
     M = length(y);
     a = (alg.outputAtLayers{3}-y);
     y_delta2 = sumsqr(a);
     err = y_delta2/2/M;
     ek = horzcat(ek,err);
     eb = 0.01;
     if err <= eb
         break;
     end
     
     %
     o_grid = (y - alg.outputAtLayers{3}) .* dlogsig(alg.outputAtLayers{3},1-alg.outputAtLayers{3});%输出层梯度
     h_grid = o_grid * alg.w{2}' .* dlogsig(alg.outputAtLayers{2},1-alg.outputAtLayers{2});
     %
     
     %更新
     dWEX = (alg.outputAtLayers{2}' * o_grid)/M;
     dB = sum(o_grid)/M;
     dwex = (alg.outputAtLayers{1}' * h_grid)/M;
     db = sum(h_grid)/M;
     %
     %
     
     alg.w{2} = alg.w{2} + lr * dWEX;
     alg.w{1} = alg.w{1} + lr * dwex;
     alg.t{2} = alg.t{2} - lr * dB;
     alg.t{1} = alg.t{1} - lr * db;

end
%[mine,index] = min(e)
figure;
plot(ek);
title('批量梯度法')
%disp(alg.outputAtLayers{3});
disp(err)
toc
end