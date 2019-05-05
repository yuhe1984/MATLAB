function MyRun(varargin)
    %clear;    
    global prob;
    warning off;
    
    if isdeployed
        runIndexStart = str2num(varargin{1});
        runIndexEnd = str2num(varargin{2});
        prob.runIndexFinish = str2num(varargin{4});
    else
        runIndexStart = varargin{1};
        runIndexEnd = varargin{2};
        prob.runIndexFinish = varargin{4};
    end
    AddPath(varargin{:});
    
    for runIndex = runIndexStart : runIndexEnd
        rand('seed', ceil(1 / (500 +1)* runIndex * (2^32)));
        randn('seed', ceil(1 / (500 +1)* runIndex * (2^32)));

        Initialize(varargin{:});
        
        Train(varargin{5});
        
        Test();
        
    end
end