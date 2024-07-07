function varargout = fixNeuronStructure(varargin)

if (size(varargin, 2) < 3 || size(varargin, 2) > 4)
    if size(varargin, 2) < 3
        cprintf('red','Error: Too few input arguments');
    else
        cprintf('red','Error: Too many input arguments');
    end
else
    inputsNo = varargin{1};
    subCubeSizes = varargin{2};
    offspringNeuronStructure = varargin{3};
    if size(varargin, 2) == 4
        offspringNeuronWeightStructure = varargin{4};
    end
    
    if sum(offspringNeuronStructure) ~= inputsNo
        difference = sum(offspringNeuronStructure) - inputsNo;
        while 1 == 1
            if difference > 0
                randomSubCubePosition = randi([1 size(offspringNeuronStructure, 2)], 1);
                randomSubCubeSize = offspringNeuronStructure(1, randomSubCubePosition);
                offspringNeuronStructure = [offspringNeuronStructure(1:randomSubCubePosition - 1) offspringNeuronStructure(randomSubCubePosition + 1:end)];
                if size(varargin, 2) == 4
                    offspringNeuronWeightStructure = [offspringNeuronWeightStructure(1:randomSubCubePosition - 1) offspringNeuronWeightStructure(randomSubCubePosition + 1:end)];
                end
                difference = difference - randomSubCubeSize;
            end
            if difference < 0
                if difference < -max(subCubeSizes)
                    randomSubCubeSize = randi([1 max(subCubeSizes)], 1);
                else
                    randomSubCubeSize = -difference;
                end
                randomSubCubePosition = randi([1 size(offspringNeuronStructure, 2) + 1], 1);
                offspringNeuronStructure = [offspringNeuronStructure(1:randomSubCubePosition - 1) randomSubCubeSize offspringNeuronStructure(randomSubCubePosition:end)];
                if size(varargin, 2) == 4
                    offspringNeuronWeightStructure = [offspringNeuronWeightStructure(1:randomSubCubePosition - 1) {rand(1, sum(2 .^ randomSubCubeSize))} offspringNeuronWeightStructure(randomSubCubePosition:end)];
                end
                difference = difference + randomSubCubeSize;
            end
            if difference == 0
                break;
            end
        end
    end
    varargout{1} = offspringNeuronStructure;
    if size(varargin, 2) == 4
        varargout{2} = offspringNeuronWeightStructure;
    end
end

