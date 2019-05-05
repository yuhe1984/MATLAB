function standardGD()
global prob alg ;
x = prob.inputData;
y = prob.outputData;
%x = mapminmax(x);
%标准梯度法
batch = length(prob.inputData);
e = [];
lr = alg.learnRate;
for i = 1:prob.runIndexFinish
e_k = [];
    for i = 1:batch
        alg.outputAtLayers{1} = x(i,:);
        total = alg.outputAtLayers{1} * alg.w{1} - alg.t{1};
        alg.outputAtLayers{2} = logsig(total); 
        totall = alg.outputAtLayers{2} * alg.w{2} - alg.t{2};
        alg.outputAtLayers{3} = logsig(totall);
        
        err = (alg.outputAtLayers{3}-y(i,:)) .* (alg.outputAtLayers{3} - y(i,:));
        y_delta2 = sum(err);
        e_kk = y_delta2/2;
        e_k = horzcat(e_k,e_kk);
        
        o_grid = zeros(1,alg.neuralN(2));
        o_grid = (y(i,:) - alg.outputAtLayers{3}) .* dlogsig(alg.outputAtLayers{3},1-alg.outputAtLayers{3});
        h_grid = zeros(1,alg.neuralN(3));
        
        a =  o_grid .* alg.w{2};
        h_grid = sum(a,2)';
        h_grid = h_grid .* dlogsig(alg.outputAtLayers{2},1-alg.outputAtLayers{2});
        
        %更新
        b = o_grid' .* alg.outputAtLayers{2};
        alg.w{2}= alg.w{2} + lr * b';
        c = h_grid' .* alg.outputAtLayers{1};
        alg.w{1} = alg.w{1} + lr * c';
        alg.t{2} = alg.t{2} - lr * o_grid;
        alg.t{1} = alg.t{1} - lr * h_grid ; 
    end
    ee = sum(e_k) / length(e_k);
    e = horzcat(e,ee);
end
%[mine,index] = min(e)
figure;
plot(e);
for i = 1:batch
        alg.outputAtLayers{1} = x(i,:);
        total = alg.outputAtLayers{1} * alg.w{1} - alg.t{1};
        alg.outputAtLayers{2} = logsig(total); 
        totall = alg.outputAtLayers{2} * alg.w{2} - alg.t{2};
        alg.outputAtLayers{3} = logsig(totall);
        alg.outputAtLayers{3};
end
title('梯度下降法')
disp(ee)
end