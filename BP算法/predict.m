function z = predict(xa)
global alg;
L = length(xa);
xa = [xa,ones(L,1)*-1];
total = xa * alg.wb;
ya = logsig(total);
hidden_input = [ya,-ones(L,1)]; 
totall = hidden_input * alg.WB;
za = logsig(totall);
% total = xa * alg.w{1} - alg.t{1};
% ya = logsig(total);
% totall = ya * alg.w{2} - alg.t{2};
% za = logsig(totall);
z = za;
m = size(z,1);
for i = 1:m
    if z(i) > 0.5
        z(i) = 1;
    elseif z(i) <= 0.5
        z(i) = 0;
    end
end
%disp(alg.outputAtLayers{3});
end