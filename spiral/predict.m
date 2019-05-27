function z = predict(xa)
global alg;

% L = length(xa);
% xa = [xa,ones(L,1)*-1];
% alg.outputAtLayers{1} = xa;
% alg.outputAtLayers{2} = tansig(alg.outputAtLayers{1} * alg.wb1);
% hidden_input1 = [alg.outputAtLayers{2},-ones(L,1)];
% alg.outputAtLayers{3} = tansig(hidden_input1 * alg.wb2);
% hidden_input2 = [alg.outputAtLayers{3},-ones(L,1)];
% alg.outputAtLayers{4} = tansig(hidden_input2 * alg.wb3);
% hidden_input3 = [alg.outputAtLayers{4},-ones(L,1)];
% alg.outputAtLayers{5} = tansig(hidden_input3 * alg.wb4);

alg.outputAtLayers{1} = xa;
hidden_input0 = alg.outputAtLayers{1} * alg.w{1} - alg.t{1};
alg.outputAtLayers{2} = tansig(hidden_input0);
hidden_input1 = alg.outputAtLayers{2} * alg.w{2} - alg.t{2};
alg.outputAtLayers{3} = tansig(hidden_input1);
hidden_input2 = alg.outputAtLayers{3} * alg.w{3} - alg.t{3};
alg.outputAtLayers{4} = tansig(hidden_input2);
hidden_input3 = alg.outputAtLayers{4} * alg.w{4} - alg.t{4};
alg.outputAtLayers{5} = tansig(hidden_input3);

z = alg.outputAtLayers{5};
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