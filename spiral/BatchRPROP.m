function BatchRPROP()
global prob alg;
x = prob.inputData;
y = prob.outputData;
%x = mapminmax(x);
%lr = alg.learnRate;
eb = [];
L = length(x);
x = [x,ones(L,1)*-1];
% default rprop parameters
delta0 = 0.1;
alg.wb1 = [alg.w{1};alg.t{1}];
alg.wb2 = [alg.w{2};alg.t{2}];
alg.wb3 = [alg.w{3};alg.t{3}];
alg.wb4 = [alg.w{4};alg.t{4}];
[m1,n1] = size(alg.wb1);
[m2,n2] = size(alg.wb2);
[m3,n3] = size(alg.wb3);
[m4,n4] = size(alg.wb4);
deltawb1 = ones(m1,n1) * delta0;
deltawb2 = ones(m2,n2) * delta0;
deltawb3 = ones(m3,n3) * delta0;
deltawb4 = ones(m4,n4) * delta0;
deltamin = 1e-6;
deltamax = 0.001;
delt_dec = 0.5;
delt_inc = 1.2;
pre_dwex1 = zeros(m1,n1);
pre_dwex2 = zeros(m2,n2);
pre_dwex3 = zeros(m3,n3);
pre_dwex4 = zeros(m4,n4);

for i = 1:prob.runIndexFinish
    alg.outputAtLayers{1} = x;
    
    alg.outputAtLayers{2} = tansig(alg.outputAtLayers{1} * alg.wb1); 
    hidden_input1 = [alg.outputAtLayers{2},-ones(L,1)];
    alg.outputAtLayers{3} = tansig(hidden_input1 * alg.wb2);
    hidden_input2 = [alg.outputAtLayers{3},-ones(L,1)];
    alg.outputAtLayers{4} = tansig(hidden_input2 * alg.wb3);
    hidden_input3 = [alg.outputAtLayers{4},-ones(L,1)];
    alg.outputAtLayers{5} = tansig(hidden_input3 * alg.wb4);
    
    M = length(y);
    a = (alg.outputAtLayers{5}-y);
    y_delta2 = sumsqr(a);
    err = y_delta2/2/M;
    eb = horzcat(eb,err);
    ebb = 0.005;
    if err < ebb
        break;
    end
    
    grid4 = (y - alg.outputAtLayers{5}) .* dtansig(hidden_input3 * alg.wb4,alg.outputAtLayers{5});%输出层梯度
    grid3 = grid4 * alg.wb4(1:end-1,:)' .* dtansig(hidden_input2 * alg.wb3,alg.outputAtLayers{4});
    grid2 = grid3 * alg.wb3(1:end-1,:)' .* dtansig(hidden_input1 * alg.wb2,alg.outputAtLayers{3});
    grid1 = grid2 * alg.wb2(1:end-1,:)' .* dtansig(alg.outputAtLayers{1} * alg.wb1,alg.outputAtLayers{2});
    dwex4 = (hidden_input3' * grid4)/M;
    dwex3 = (hidden_input2' * grid3)/M;
    dwex2 = (hidden_input1' * grid2)/M;
    dwex1 = (alg.outputAtLayers{1}' * grid1)/M;
    ggwb4 = dwex4 .* pre_dwex4;
    %size(hidden_input2)
    ggwb3 = dwex3 .* pre_dwex3;
    ggwb2 = dwex2 .* pre_dwex2;
    ggwb1 = dwex1 .* pre_dwex1;
%     dw4 = (alg.outputAtLayers{4}' * grid4)/M;
%     dw3 = (alg.outputAtLayers{3}' * grid3)/M;
%     dw2 = (alg.outputAtLayers{2}' * grid2)/M;
%     dw1 = (x(:,1:end-1)' * grid1)/M;
%     db4 = sum(grid4)/M;
%     db3 = sum(grid3)/M;
%     db2 = sum(grid2)/M;
%     db1 = sum(grid1)/M;
%     dwex4 = [dw4;db4];
%     dwex3 = [dw3;db3];
%     dwex2 = [dw2;db2];
%     dwex1 = [dw1;db1];
    for t = 1:m4
        for k = 1:n4
            if ggwb4(t,k) > 0
                deltawb4(t,k) = min(deltawb4(t,k)*delt_inc,deltamax);
                dwex4(t,k) = sign(dwex4(t,k)) * deltawb4(t,k);
                alg.wb4(t,k) = alg.wb4(t,k) + dwex4(t,k);
            elseif ggwb4(t,k) < 0
                deltawb4(t,k) = max(deltawb4(t,k)*delt_dec,deltamin);
                alg.wb4(t,k) = alg.wb4(t,k) - pre_dwex4(t,k);
                dwex4(t,k) = 0;
            elseif ggwb4(t,k) == 0
                dwex4(t,k) = sign(dwex4(t,k)) * deltawb4(t,k);
                alg.wb4(t,k) = alg.wb4(t,k) + dwex4(t,k);
            end
        end
    end
    for t = 1:m3
        for k = 1:n3
            if ggwb3(t,k) > 0
                deltawb3(t,k) = min(deltawb3(t,k)*delt_inc,deltamax);
                dwex3(t,k) = sign(dwex3(t,k)) * deltawb3(t,k);
                alg.wb3(t,k) = alg.wb3(t,k) + dwex3(t,k);
            elseif ggwb3(t,k) < 0
                deltawb3(t,k) = max(deltawb3(t,k)*delt_dec,deltamin);
                alg.wb3(t,k) = alg.wb3(t,k) - pre_dwex3(t,k);
                dwex3(t,k) = 0;
            elseif ggwb3(t,k) == 0
                dwex3(t,k) = sign(dwex3(t,k)) * deltawb3(t,k);
                alg.wb3(t,k) = alg.wb3(t,k) + dwex3(t,k);
            end
        end
    end
    for t = 1:m2
        for k = 1:n2
            if ggwb2(t,k) > 0
                deltawb2(t,k) = min(deltawb2(t,k)*delt_inc,deltamax);
                dwex2(t,k) = sign(dwex2(t,k)) * deltawb2(t,k);
                alg.wb2(t,k) = alg.wb2(t,k) + dwex2(t,k);
            elseif ggwb2(t,k) < 0
                deltawb2(t,k) = max(deltawb2(t,k)*delt_dec,deltamin);
                alg.wb2(t,k) = alg.wb2(t,k) - pre_dwex2(t,k);
                dwex2(t,k) = 0;
            elseif ggwb2(t,k) == 0
                dwex2(t,k) = sign(dwex2(t,k)) * deltawb2(t,k);
                alg.wb2(t,k) = alg.wb2(t,k) + dwex2(t,k);
            end
        end
    end
    for t = 1:m1
        for k = 1:n1
            if ggwb1(t,k) > 0
                deltawb1(t,k) = min(deltawb1(t,k)*delt_inc,deltamax);
                dwex1(t,k) = sign(dwex1(t,k)) * deltawb1(t,k);
                alg.wb1(t,k) = alg.wb1(t,k) + dwex1(t,k);
            elseif ggwb1(t,k) < 0
                deltawb1(t,k) = max(deltawb1(t,k)*delt_dec,deltamin);
                alg.wb1(t,k) = alg.wb1(t,k) - pre_dwex1(t,k);
                dwex1(t,k) = 0;
            elseif ggwb1(t,k) == 0
                dwex1(t,k) = sign(dwex1(t,k)) * deltawb1(t,k);
                alg.wb1(t,k) = alg.wb1(t,k) + dwex1(t,k);
            end
        end
    end
    pre_dwex4 = dwex4;
    pre_dwex3 = dwex3;
    pre_dwex2 = dwex2;
    pre_dwex1 = dwex1;
    %alg.t{2} = alg.t{2} - lr * dB;
    %alg.t{1} = alg.t{1} - lr * db;
end
%[mine,index] = min(e)
huatu
figure;
plot(eb);
disp(err)
%disp(i)
%[alg.outputAtLayers{5},y];
title('批量弹性算法')
end