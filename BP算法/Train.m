function Train(k)

%�ݶ��½���
if k == 1
    standardGD();
elseif k == 2
    BatchGD();
%�����㷨
elseif k == 3
    standardRPROP();
elseif k == 4
    BatchRPROP();
%������
elseif k == 5
    standardMC();
elseif k == 6
    BatchMC();
%����Ӧѧϰ��
elseif k == 7
    standardLR();
elseif k == 8
    BatchLR();
%������������Ӧѧϰ��
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