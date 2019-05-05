function Train(k)

%梯度下降法
if k == 1
    standardGD();
elseif k == 2
    BatchGD();
%弹性算法
elseif k == 3
    standardRPROP();
elseif k == 4
    BatchRPROP();
%动量法
elseif k == 5
    standardMC();
elseif k == 6
    BatchMC();
%自适应学习率
elseif k == 7
    standardLR();
elseif k == 8
    BatchLR();
%带动量的自适应学习率
elseif k == 9
    standardlrMC();
elseif k == 10
    BatchlrMC();
%
elseif k == 11
    Dynamic();
elseif k == 12
    BatchRPROPtest();
end

end