function BatchMC()
global prob alg ;
x = prob.inputData;
y = prob.outputData;
%x = mapminmax(x);
eb = [];
lr = alg.learnRate;
mc = 0.9;
[m,n] = size(alg.w{2});
[mm,nn] = size(alg.w{1});
pre_dWEX = zeros(m,n);
pre_dwex = zeros(mm,nn);
pre_dB = zeros(1,n);
pre_db = zeros(1,nn);
for i = 1:prob.runIndexFinish
    alg.outputAtLayers{1} = x;
    
    total = alg.outputAtLayers{1} * alg.w{1} - alg.t{1};
    alg.outputAtLayers{2} = logsig(total);
    totall = alg.outputAtLayers{2} * alg.w{2} - alg.t{2};
    alg.outputAtLayers{3} = logsig(totall);
    
    M = length(y);
    a = (alg.outputAtLayers{3}-y);
    y_delta2 = sumsqr(a);
    err = y_delta2/2/M;
    eb = horzcat(eb,err);
    ebb = 0.005;
    if err < ebb
        break;
    end
    
    o_grid = (y - alg.outputAtLayers{3}) .* dlogsig(alg.outputAtLayers{3},1-alg.outputAtLayers{3});%输出层梯度
    h_grid = o_grid * alg.w{2}' .* dlogsig(alg.outputAtLayers{2},1-alg.outputAtLayers{2});
    %
    
    %更新
    dWEX = (alg.outputAtLayers{2}' * o_grid)/M;
    dB = sum(o_grid)/M;
    dwex = (alg.outputAtLayers{1}' * h_grid)/M;
    db = sum(h_grid)/M;
    alg.w{2} = alg.w{2} + lr * (1-mc) * dWEX + mc * pre_dWEX;
    alg.w{1} = alg.w{1} + lr * (1-mc) * dwex + mc * pre_dwex;
    alg.t{2} = alg.t{2} - lr * (1-mc) * dB - mc * pre_dB;
    alg.t{1} = alg.t{1} - lr * (1-mc) * db - mc * pre_db;
    pre_dWEX = dWEX;
    pre_dB = dB;
    pre_dwex = dwex;
    pre_db = db;
end
%[mine,index] = min(e)
figure;
plot(eb);
title('批量动量梯度下降法')
%disp(alg.outputAtLayers{3});
%disp(err)
disp(i)
end