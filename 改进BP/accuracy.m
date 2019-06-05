function accuracy()
global prob alg;
x = prob.inputData;
y = prob.outputData;
alg.outputAtLayers{1} = x;
    for j = 2:numel(alg.outputAtLayers)
        alg.outputAtLayers{j} = logsig(alg.outputAtLayers{j-1} * alg.w{j-1} - alg.t{j-1});
    end
Y = alg.outputAtLayers{numel(alg.outputAtLayers)};
%统计识别正确率
[s1 , s2] = size(Y) ;
hitNum = 0 ;
if s2 == 1
    for i = 1:s1
        if Y(i) > 0.5
            Y(i) = 1;
        elseif Y(i) <= 0.5
            Y(i) = 0;
        end
        if Y(i) == y(i)
            hitNum = hitNum + 1;
        end
    end
else
    for i = 1 : s1
        [~ , Index1] = max( Y( i,: ) ) ;
        [~ , Index2] = max( y( i,: ) ) ;
        if( Index1 == Index2   )
            hitNum = hitNum + 1 ;
        end
    end
end
sprintf('识别率是 %3.3f%%',100 * hitNum / s1 )
end