function BatchRPROP()
global prob alg;
x = prob.inputData;
y = prob.outputData;
%x = mapminmax(x);
%lr = alg.learnRate;
eb = [];

% default rprop parameters
delta0 = 0.1;
[m,n] = size(alg.w{2});
[mm,nn] = size(alg.w{1});
deltaWB = ones(m,n) * delta0;
deltawb = ones(mm,nn) * delta0;
deltaB = ones(1,n) * delta0;
deltab = ones(1,nn) * delta0;
deltamin = 1e-6;
deltamax = 50;
delt_dec = 0.5;
delt_inc = 1.2;
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
    ebb = 0.01;
    if err < ebb
        break;
    end
    
    o_grid = (y - alg.outputAtLayers{3}) .* dlogsig(alg.outputAtLayers{3},1-alg.outputAtLayers{3});%输出层梯度
    h_grid = o_grid * alg.w{2}' .* dlogsig(alg.outputAtLayers{2},1-alg.outputAtLayers{2});
    dWEX = (alg.outputAtLayers{2}' * o_grid)/M;
    dB = sum(o_grid)/M;
    dwex = (alg.outputAtLayers{1}' * h_grid)/M;
    db = sum(h_grid)/M;
    ggWB = dWEX .* pre_dWEX;
    ggwb = dwex .* pre_dwex;
    ggB = dB .* pre_dB;
    ggb = db .* pre_db;
    for t = 1:m
        for k = 1:n
            if ggWB(t,k) > 0
                deltaWB(t,k) = min(deltaWB(t,k)*delt_inc,deltamax);
                dWEX(t,k) = sign(dWEX(t,k)) * deltaWB(t,k);
                alg.w{2}(t,k) = alg.w{2}(t,k) + dWEX(t,k);
            elseif ggWB(t,k) < 0
                deltaWB(t,k) = max(deltaWB(t,k)*delt_dec,deltamin);
                alg.w{2}(t,k) = alg.w{2}(t,k) - pre_dWEX(t,k);
                dWEX(t,k) = 0;
            elseif ggWB(t,k) == 0
                dWEX(t,k) = sign(dWEX(t,k)) * deltaWB(t,k);
                alg.w{2}(t,k) = alg.w{2}(t,k) + dWEX(t,k);
            end
        end
    end
    for t = 1:n
        if ggB(1,t) > 0
            deltaB(1,t) = min(deltaB(1,t)*delt_inc,deltamax);
            dB(1,t) = sign(dB(1,t)) * deltaB(1,t);
            alg.t{2}(1,t) = alg.t{2}(1,t) + dB(1,t);
        elseif ggB(1,t) < 0
            deltaB(1,t) = max(deltaB(1,t)*delt_dec,deltamin);
            alg.t{2}(1,t) = alg.t{2}(1,t) - pre_dB(1,t);
            dB(1,t) = 0;
        elseif ggB(1,t) == 0
            dB(1,t) = sign(dB(1,t)) * deltaB(1,t);
            alg.t{2}(1,t) = alg.t{2}(1,t) + dB(1,t);
        end
    end
    for t = 1:mm
        for k = 1:nn
            if ggwb(t,k) > 0
                deltawb(t,k) = min(deltawb(t,k)*delt_inc,deltamax);
                dwex(t,k) = sign(dwex(t,k)) * deltawb(t,k);
                alg.w{1}(t,k) = alg.w{1}(t,k) + dwex(t,k);
            elseif ggwb(t,k) < 0
                deltawb(t,k) = max(deltawb(t,k)*delt_dec,deltamin);
                alg.w{1}(t,k) = alg.w{1}(t,k) - pre_dwex(t,k);
                dwex(t,k) = 0;
            elseif ggwb(t,k) == 0
                dwex(t,k) = sign(dwex(t,k)) * deltawb(t,k);
                alg.w{1}(t,k) = alg.w{1}(t,k) + dwex(t,k);
            end
        end
    end
    for t = 1:nn
        if ggb(1,t) > 0
            deltab(1,t) = min(deltab(1,t)*delt_inc,deltamax);
            db(1,t) = sign(db(1,t)) * deltab(1,t);
            alg.t{1}(1,t) = alg.t{1}(1,t) + db(1,t);
        elseif ggb(1,t) < 0
            deltab(1,t) = max(deltab(1,t)*delt_dec,deltamin);
            alg.t{1}(1,t) = alg.t{1}(1,t) - pre_db(1,t);
            db(1,t) = 0;
        elseif ggb(1,t) == 0
            db(1,t) = sign(db(1,t)) * deltab(1,t);
            alg.t{1}(1,t) = alg.t{1}(1,t) + db(1,t);
        end
    end
    pre_dWEX = dWEX;
    pre_dB = dB;
    pre_dwex = dwex;
    pre_db = db;
    %alg.t{2} = alg.t{2} - lr * dB;
    %alg.t{1} = alg.t{1} - lr * db;
end
%[mine,index] = min(e)
figure;
plot(eb);
disp(err)
alg.outputAtLayers{3};
title('批量弹性算法')
end