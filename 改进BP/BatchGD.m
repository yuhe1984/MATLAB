function BatchGD()
%tic
global prob alg;
x = prob.inputData;
y = prob.outputData;
%x = mapminmax(x);
lr = alg.learnRate;
ek = [];
M = length(y);
eb = 0.005;
for i = 1:prob.runIndexFinish
    %forward
    alg.outputAtLayers{1} = x;
    for j = 2:numel(alg.outputAtLayers)
        alg.outputAtLayers{j} = logsig(alg.outputAtLayers{j-1} * alg.w{j-1} - alg.t{j-1});
    end
    
    err = sumsqr(alg.outputAtLayers{3}-y)/2/M;
    ek = [ek,err];
    if err <= eb
        break;
    end
    
    %back propagation
    grid{2} = (y - alg.outputAtLayers{3}) .* dlogsig(alg.outputAtLayers{3},1-alg.outputAtLayers{3});
    grid{1} = grid{2} * alg.w{2}' .* dlogsig(alg.outputAtLayers{2},1-alg.outputAtLayers{2});
    
    %更新
    %      dw{2} = (alg.outputAtLayers{2}' * grid{2})/M;
    %      dt{2} = sum(grid{2})/M;
    %      dw{1} = (alg.outputAtLayers{1}' * grid{1})/M;
    %      dt{1} = sum(grid{1})/M;
    
    for z = 1:alg.layersN-1
        dw{z} = (alg.outputAtLayers{z}' * grid{z})/M;
        dt{z} = sum(grid{z})/M;
    end
    
    for k = 1:alg.layersN-1
        alg.w{k} = alg.w{k} + lr * dw{k};
        alg.t{k} = alg.t{k} - lr * dt{k};
    end
     
end
%[mine,index] = min(e)
figure;
plot(ek);
title('批量梯度法')
%disp([alg.outputAtLayers{3},y]);
disp(err)
disp(i)
%huatu
%toc
end