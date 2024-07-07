function population = createInitialPopulation(inputsNo, outputsNo, hiddenLayerNeuronsNo, subCubeSizes, populationSize, inputRange)

neuronsNo = hiddenLayerNeuronsNo + outputsNo;
population = cell(populationSize, neuronsNo);

for currentPopulation = 1:populationSize
    parfor currentNeuron = 1:hiddenLayerNeuronsNo
        for currentHiddenNeuron = 1:hiddenLayerNeuronsNo
            neuron = struct('hiddenLayerNeuronStructure', [], 'hiddenLayerWeights', []);
            neuron.hiddenLayerNeuronStructure = createMultiCubeNeuronStructure(inputsNo, subCubeSizes);
            subCubesNo = size(neuron.hiddenLayerNeuronStructure, 2);
            neuron.hiddenLayerWeights = cell(1, subCubesNo);
            for currentSubCube = 1:subCubesNo
                neuron.hiddenLayerWeights{1, currentSubCube} = inputRange(1) + (inputRange(2) - inputRange(1)) .* rand(1, sum(2 .^ neuron.hiddenLayerNeuronStructure(1, currentSubCube)));
            end
            population{currentPopulation, currentNeuron} = neuron;
        end
    end
    parfor currentOutputNeuron = hiddenLayerNeuronsNo + 1:neuronsNo
        neuron = struct('outputLayerNeuronStructure', []);
        neuron.outputLayerNeuronStructure = createMultiCubeNeuronStructure(hiddenLayerNeuronsNo, subCubeSizes);
        population{currentPopulation, currentOutputNeuron} = neuron;
    end
end
end

