function varargout = exchangeData(varargin)

if (size(varargin, 2) < 2 || size(varargin, 2) > 4)
    if size(varargin, 2) < 2
        cprintf('red','Error: Too few input arguments');
    else
        cprintf('red','Error: Too many input arguments');
    end
else
    varargout{1} = varargin{1};
    varargout{2} = varargin{2};
    
    temp = varargout{2};
    varargout{2} = varargout{1};
    varargout{1} = temp;
    
    if size(varargin, 2) == 4
        varargout{3} = varargin{3};
        varargout{4} = varargin{4};
        
        temp = varargout{4};
        varargout{4} = varargout{3};
        varargout{3} = temp;
    end
end
end

