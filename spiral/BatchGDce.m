function BatchGDce()
global prob alg ;
x = prob.inputData;
y = prob.outputData;
%x = mapminmax(x);
eb = [];
lr = alg.learnRate;
for i = 1:prob.runIndexFinish    
    alg.outputAtLayers{1} = x;
    hidden_input0 = alg.outputAtLayers{1} * alg.w{1} - alg.t{1};
    alg.outputAtLayers{2} = logsig(hidden_input0); 
    hidden_input1 = alg.outputAtLayers{2} * alg.w{2} - alg.t{2};
    alg.outputAtLayers{3} = logsig(hidden_input1);
    hidden_input2 = alg.outputAtLayers{3} * alg.w{3} - alg.t{3};
    alg.outputAtLayers{4} = logsig(hidden_input2);
    hidden_input3 = alg.outputAtLayers{4} * alg.w{4} - alg.t{4};
    alg.outputAtLayers{5} = logsig(hidden_input3);
    
    M = length(y);
    a = (-y.*log(alg.outputAtLayers{5})-(1-y).*log(1-alg.outputAtLayers{5}));
    err = sum(a)/M;
    eb = horzcat(eb,err);
    ebb = 0.005;
    if err < ebb
        break;
    end
    
    grid4 = (y - alg.outputAtLayers{5});%输出层梯度
    grid3 = grid4 * alg.w{4}';
    grid2 = grid3 * alg.w{3}';
    grid1 = grid2 * alg.w{2}';
    dwex4 = (alg.outputAtLayers{4}' * grid4)/M;
    dwex3 = (alg.outputAtLayers{3}' * grid3)/M;
    dwex2 = (alg.outputAtLayers{2}' * grid2)/M;
    dwex1 = (alg.outputAtLayers{1}' * grid1)/M;
    db4 = sum(grid4)/M;
    db3 = sum(grid3)/M;
    db2 = sum(grid2)/M;
    db1 = sum(grid1)/M;
    %
    
    %更新
    alg.w{4} = alg.w{4} + lr * dwex4;
    alg.w{3} = alg.w{3} + lr * dwex3;
    alg.w{2} = alg.w{2} + lr * dwex2;
    alg.w{1} = alg.w{1} + lr * dwex1;
    alg.t{4} = alg.t{4} - lr * db4;
    alg.t{3} = alg.t{3} - lr * db3;
    alg.t{2} = alg.t{2} - lr * db2;
    alg.t{1} = alg.t{1} - lr * db1;
end
%[mine,index] = min(e)
figure;
plot(eb);
title('批量动量梯度下降法')
%disp([alg.outputAtLayers{5},y]);
%disp(err)
huatu
disp(i)
end