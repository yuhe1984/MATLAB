function Train(k)

if k == 1
    BatchRPROP();
elseif k == 2
    BatchGD();
elseif k == 3
    BatchGDce();
end

end