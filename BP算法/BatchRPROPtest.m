function BatchRPROPtest()
global prob alg;
x = prob.inputData;
y = prob.outputData;
x = mapminmax(x);
%lr = alg.learnRate;
eb = [];
L = length(x);
x = [x,ones(L,1)*-1];
% default rprop parameters
delta0 = 0.1;
WB = [alg.w{2};alg.t{2}];
wb = [alg.w{1};alg.t{1}];
[m,n] = size(WB);
[mm,nn] = size(wb);
deltaWB = ones(m,n) * delta0;
deltawb = ones(mm,nn) * delta0;
deltamin = 1e-6;
deltamax = 50;
delt_dec = 0.5;
delt_inc = 1.2;
pre_dWEX = zeros(m,n);
pre_dwex = zeros(mm,nn); 

for i = 1:prob.runIndexFinish
    alg.outputAtLayers{1} = x;
    
    alg.outputAtLayers{2} = logsig(alg.outputAtLayers{1} * wb); 
    hidden_input = [alg.outputAtLayers{2},-ones(L,1)];
    alg.outputAtLayers{3} = logsig(hidden_input * WB);
    
    M = length(y);
    a = (alg.outputAtLayers{3}-y);
    y_delta2 = sumsqr(a);
    err = y_delta2/2/M;
    eb = horzcat(eb,err);
    ebb = 0.01;
    if err < ebb
        break;
    end
    
    o_grid = (y - alg.outputAtLayers{3}) .* dlogsig(alg.outputAtLayers{3},1-alg.outputAtLayers{3});%输出层梯度
    h_grid = o_grid * WB(1:end-1,:)' .* dlogsig(alg.outputAtLayers{2},1-alg.outputAtLayers{2});
    dWEX = (hidden_input' * o_grid)/M;
    dwex = (alg.outputAtLayers{1}' * h_grid)/M;
    ggWB = dWEX .* pre_dWEX;
    ggwb = dwex .* pre_dwex;
    for t = 1:m
        for k = 1:n
            if ggWB(t,k) > 0
                deltaWB(t,k) = min(deltaWB(t,k)*delt_inc,deltamax);
                dWEX(t,k) = sign(dWEX(t,k)) * deltaWB(t,k);
                WB(t,k) = WB(t,k) + dWEX(t,k);
            elseif ggWB(t,k) < 0
                deltaWB(t,k) = max(deltaWB(t,k)*delt_dec,deltamin);
                WB(t,k) = WB(t,k) - pre_dWEX(t,k);
                dWEX(t,k) = 0;
            elseif ggWB(t,k) == 0
                dWEX(t,k) = sign(dWEX(t,k)) * deltaWB(t,k);
                WB(t,k) = WB(t,k) + dWEX(t,k);
            end
        end
    end
    for t = 1:mm
        for k = 1:nn
            if ggwb(t,k) > 0
                deltawb(t,k) = min(deltawb(t,k)*delt_inc,deltamax);
                dwex(t,k) = sign(dwex(t,k)) * deltawb(t,k);
                wb(t,k) = wb(t,k) + dwex(t,k);
            elseif ggwb(t,k) < 0
                deltawb(t,k) = max(deltawb(t,k)*delt_dec,deltamin);
                wb(t,k) = wb(t,k) - pre_dwex(t,k);
                dwex(t,k) = 0;
            elseif ggwb(t,k) == 0
                dwex(t,k) = sign(dwex(t,k)) * deltawb(t,k);
                wb(t,k) = wb(t,k) + dwex(t,k);
            end
        end
    end
    pre_dWEX = dWEX;
    pre_dwex = dwex;
    %alg.t{2} = alg.t{2} - lr * dB;
    %alg.t{1} = alg.t{1} - lr * db;
end
%[mine,index] = min(e)
figure;
plot(eb);
disp(err)
[alg.outputAtLayers{3},y]
title('批量弹性算法')
end