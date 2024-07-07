function customNeuron = createMultiCubeNeuronStructure(inputsNo, subCubeSizes)
neuronSize = inputsNo;
subCubeSize = subCubeSizes;
customNeuron = [];
while true
    while true
        if neuronSize < max(subCubeSize)
            subCubeSize = subCubeSize(1, 1:end -1);
        else
            break;
        end
    end
    selectRandomSubCube = randperm(size(subCubeSize, 2), 1);
    customNeuron = [customNeuron subCubeSize(selectRandomSubCube)];
    neuronSize = neuronSize - subCubeSize(selectRandomSubCube);
    if subCubeSize(selectRandomSubCube) == 0
        neuronSize = neuronSize - 1;
    end
    if neuronSize == 0
        break;
    end
end
end

