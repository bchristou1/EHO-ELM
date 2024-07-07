function offspring = crossover(hiddenLayerNeuronsNo, population, subCubeSizes)

chromosomesNo = size(population, 1);
chromosomeLength = size(population, 2);
chromosomePairs = floor(chromosomesNo / 2);
offspring = cell(2 * chromosomePairs, chromosomeLength);
inputsNo = sum(population{1, 1}.hiddenLayerNeuronStructure);

for currentChromosomePair = 1:chromosomePairs
    parentOneLocation =  2 * currentChromosomePair - 1;
    parentTwoLocation =  2 * currentChromosomePair;
    parentOne = population(parentOneLocation, :);
    parentTwo = population(parentTwoLocation, :);
    
    for currentNeuron = 1:chromosomeLength
        if currentNeuron <= hiddenLayerNeuronsNo
            parentOneCurrentNeuronStructure = parentOne{1, currentNeuron}.hiddenLayerNeuronStructure;
            parentTwoCurrentNeuronStructure = parentTwo{1, currentNeuron}.hiddenLayerNeuronStructure;
            parentOneCurrentNeuronWeightStructure = parentOne{1, currentNeuron}.hiddenLayerWeights;
            parentTwoCurrentNeuronWeightStructure = parentTwo{1, currentNeuron}.hiddenLayerWeights;
        else
            parentOneCurrentNeuronStructure = parentOne{1, currentNeuron}.outputLayerNeuronStructure;
            parentTwoCurrentNeuronStructure = parentTwo{1, currentNeuron}.outputLayerNeuronStructure;
        end
        
        parentOneCurrentNeuronStructureSize = size(parentOneCurrentNeuronStructure, 2);
        parentTwoCurrentNeuronStructureSize = size(parentTwoCurrentNeuronStructure, 2);
        
        offspringOneCurrentNeuronStructure = parentOneCurrentNeuronStructure;
        offspringTwoCurrentNeuronStructure = parentTwoCurrentNeuronStructure;
        offspringOneCurrentNeuronWeightStructure = parentOneCurrentNeuronWeightStructure;
        offspringTwoCurrentNeuronWeightStructure = parentTwoCurrentNeuronWeightStructure;
        
        if parentOneCurrentNeuronStructureSize < parentTwoCurrentNeuronStructureSize
            while 1 == 1
                mask = randi([0 1],[1 parentOneCurrentNeuronStructureSize]);
                if sum(mask) > 0
                    break;
                end
            end
        else
            while 1 == 1
                mask = randi([0 1],[1 parentTwoCurrentNeuronStructureSize]);
                if sum(mask) > 0
                    break;
                end
            end
        end
        
        for count = 1:size(mask, 2)
            if mask(1, count) == 1
                if currentNeuron <= hiddenLayerNeuronsNo
                    [offspringOneCurrentNeuronStructure(1, count), offspringTwoCurrentNeuronStructure(1, count), offspringOneCurrentNeuronWeightStructure(1, count), offspringTwoCurrentNeuronWeightStructure(1, count)] = ...
                        exchangeData(offspringOneCurrentNeuronStructure(1, count), offspringTwoCurrentNeuronStructure(1, count), offspringOneCurrentNeuronWeightStructure(1, count), offspringTwoCurrentNeuronWeightStructure(1, count));
                else
                    [offspringOneCurrentNeuronStructure(1, count), offspringTwoCurrentNeuronStructure(1, count)] = exchangeData(offspringOneCurrentNeuronStructure(1, count), offspringTwoCurrentNeuronStructure(1, count));
                end
            end
        end
        
        if currentNeuron <= hiddenLayerNeuronsNo
            neuron = struct('hiddenLayerNeuronStructure', [], 'hiddenLayerWeights', []);
            [neuron.hiddenLayerNeuronStructure, neuron.hiddenLayerWeights] = fixNeuronStructure(inputsNo, subCubeSizes, offspringOneCurrentNeuronStructure, offspringOneCurrentNeuronWeightStructure);
            offspring{parentOneLocation, currentNeuron} = neuron;
            neuron = struct('hiddenLayerNeuronStructure', [], 'hiddenLayerWeights', []);
            [neuron.hiddenLayerNeuronStructure, neuron.hiddenLayerWeights] = fixNeuronStructure(inputsNo, subCubeSizes, offspringTwoCurrentNeuronStructure, offspringTwoCurrentNeuronWeightStructure);
            offspring{parentTwoLocation, currentNeuron} = neuron;
        else
            neuron = struct('outputLayerNeuronStructure', []);
            neuron.outputLayerNeuronStructure = fixNeuronStructure(hiddenLayerNeuronsNo, subCubeSizes, offspringOneCurrentNeuronStructure);
            offspring{parentOneLocation, currentNeuron} = neuron;
            neuron = struct('outputLayerNeuronStructure', []);
            neuron.outputLayerNeuronStructure = fixNeuronStructure(hiddenLayerNeuronsNo, subCubeSizes, offspringTwoCurrentNeuronStructure);
            offspring{parentTwoLocation, currentNeuron} = neuron;
        end
    end
end

