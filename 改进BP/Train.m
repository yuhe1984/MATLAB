function Train(k)

%�ݶ��½���
if k == 1
    BatchGD();
elseif k == 2
    DynamicBGD();

end