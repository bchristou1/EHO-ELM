function population =  mutation(population, offspring, mutationRate, hiddenLayerNeuronsNo, inputRange)

offspringChromosomesNo = size(offspring, 1);
offspringMutatedChromosomesNo = round(mutationRate * offspringChromosomesNo);
offspringMutatedChromosomes = randperm(offspringChromosomesNo, offspringMutatedChromosomesNo);

for currentMutatedChromosome = 1:offspringMutatedChromosomesNo
    mutatedNeuron = randperm(hiddenLayerNeuronsNo, 1);
    offspringCurrentNeuronStructure = offspring{offspringMutatedChromosomes(1, currentMutatedChromosome), mutatedNeuron}.hiddenLayerNeuronStructure;
    subCubesNo = size(offspringCurrentNeuronStructure, 2);
    mutatedSubCube = randperm(subCubesNo, 1);
    offspring{offspringMutatedChromosomes(1, currentMutatedChromosome), mutatedNeuron}.hiddenLayerWeights{1, mutatedSubCube} = inputRange(1) + (inputRange(2) - inputRange(1)) .* rand(1, sum(2 .^ offspringCurrentNeuronStructure(mutatedSubCube)));
end

population = [population; offspring]; 

end

