function Initialize(varargin) 
    global prob alg;
    
    dataFilePathName = varargin{3};
    
    data = load(dataFilePathName);
    prob.inputData = data(:,1: end-1);
    prob.outputData = data(:,end);
    if max(prob.outputData) ~= 1
        outputData = data(:, end) - min(data(:, end)) + 1;
        prob.outputData = zeros(size(data,1), max(outputData));
        %prob.outputData(:, outputData) = 1;
        for i = 1:length(prob.outputData)
            prob.outputData(i,outputData(i)) = 1;
        end
    end
    
    dataN = size(prob.inputData,1);
    
    %
    prob.runIndexFinish = varargin{4};
    randTrainIndex = randperm(dataN, ceil(0.5*dataN));
    prob.trainLogical = false(dataN, 1);
    prob.trainLogical(randTrainIndex) = true;    
    prob.testLogical = true(size(prob.inputData, 1));
    prob.testLogical(randTrainIndex) = false;
    %
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    alg.learnRate = 0.1;
    alg.neuralN = [size(prob.inputData, 2), ...
                            5, ...
                            size(prob.outputData, 2)];
    alg.layersN = numel(alg.neuralN);
    alg.outputAtLayers = cell(1, alg.layersN);
    for i = 1 : numel(alg.outputAtLayers)
        alg.outputAtLayers(i) = {zeros(1,alg.neuralN(i))};
    end
    
    alg.w = cell(1, alg.layersN-1);
    for i = 1 : numel(alg.w)
        alg.w(i) = {rand(alg.neuralN(i),  alg.neuralN(i+1))};
    end
    
    alg.t = cell(1, alg.layersN-1);
    for i = 1 : numel(alg.t)
        alg.t(i) = {rand(1,alg.neuralN(i+1))};
    end    
    
    alg.f = @(x) 1/(1+exp(-x));
    alg.df = @(y) y .* (1 -y);
end