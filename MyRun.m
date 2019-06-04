function MyRun(varargin)
    %clear;    
    warning off;
    
    if isdeployed
        runIndexStart = str2num(varargin{1});
        runIndexEnd = str2num(varargin{2});
    else
        runIndexStart = varargin{1};
        runIndexEnd = varargin{2};
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