function varargout = selection(validationResultsFolds, population, offspring, populationSize, outputsNo, inputTrain, inputValidation, inputTest, outputTrain, outputValidation, outputTest, transferFunction, testSet, datasetType, multiCoreCPU, keepNetworkOutputs)

foldsNo = size(outputValidation, 2);

if isempty(validationResultsFolds)
    validationResultsFolds = zeros(foldsNo, size(population, 1));
    if keepNetworkOutputs == 1
        networkOutputsTable = cell(foldsNo, size(population, 1));
    end
    if testSet == 0
        inputTest = [];
        outputTest = [];
        for currentChromosome = 1:size(population, 1)
            if multiCoreCPU == 1
                validationResultsFolds(:, currentChromosome) = multiCubeELMParallel(outputsNo, inputTrain, inputValidation, inputTest, outputTrain, outputValidation, outputTest, population(currentChromosome, :), transferFunction, testSet, datasetType, keepNetworkOutputs);
            else
                validationResultsFolds(:, currentChromosome) = multiCubeELM(outputsNo, inputTrain, inputValidation, inputTest, outputTrain, outputValidation, outputTest, population(currentChromosome, :), transferFunction, testSet, datasetType, keepNetworkOutputs);
            end
        end
    else
        testResultsFolds = zeros(foldsNo, size(population, 1));
        for currentChromosome = 1:size(population, 1)
            if multiCoreCPU == 1
                if keepNetworkOutputs == 1
                    [validationResultsFolds(:, currentChromosome), testResultsFolds(:, currentChromosome), networkOutputsTable(:, currentChromosome)] = multiCubeELMParallel(outputsNo, inputTrain, inputValidation, inputTest, outputTrain, outputValidation, outputTest, population(currentChromosome, :), transferFunction, testSet, datasetType, keepNetworkOutputs);
                else
                    [validationResultsFolds(:, currentChromosome), testResultsFolds(:, currentChromosome)] = multiCubeELMParallel(outputsNo, inputTrain, inputValidation, inputTest, outputTrain, outputValidation, outputTest, population(currentChromosome, :), transferFunction, testSet, datasetType, keepNetworkOutputs);
                end
            else
                if keepNetworkOutputs == 1
                    [validationResultsFolds(:, currentChromosome), testResultsFolds(:, currentChromosome), networkOutputsTable(:, currentChromosome)] = multiCubeELM(outputsNo, inputTrain, inputValidation, inputTest, outputTrain, outputValidation, outputTest, population(currentChromosome, :), transferFunction, testSet, datasetType, keepNetworkOutputs);
                else
                    [validationResultsFolds(:, currentChromosome), testResultsFolds(:, currentChromosome)] = multiCubeELM(outputsNo, inputTrain, inputValidation, inputTest, outputTrain, outputValidation, outputTest, population(currentChromosome, :), transferFunction, testSet, datasetType, keepNetworkOutputs);
                end
            end
        end
    end
else
    validationResultsFolds = [validationResultsFolds zeros(foldsNo, size(offspring, 1))];
    if testSet == 0
        for currentChromosome = 1:size(offspring, 1)
            if multiCoreCPU == 1
                validationResultsFolds(:, populationSize + currentChromosome) = multiCubeELMParallel(outputsNo, inputTrain, inputValidation, inputTest, outputTrain, outputValidation, outputTest, offspring(currentChromosome, :), transferFunction, testSet, datasetType, keepNetworkOutputs);
            else
                validationResultsFolds(:, populationSize + currentChromosome) = multiCubeELM(outputsNo, inputTrain, inputValidation, inputTest, outputTrain, outputValidation, outputTest, offspring(currentChromosome, :), transferFunction, testSet, datasetType, keepNetworkOutputs);
            end
        end
    else
        testResultsFolds = zeros(foldsNo, size(population, 1));
        for currentChromosome = 1:size(population, 1)
            if multiCoreCPU == 1
                if keepNetworkOutputs == 1
                    [validationResultsFolds(:, currentChromosome), testResultsFolds(:, currentChromosome), networkOutputsTable(:, currentChromosome)] = multiCubeELMParallel(outputsNo, inputTrain, inputValidation, inputTest, outputTrain, outputValidation, outputTest, population(currentChromosome, :), transferFunction, testSet, datasetType, keepNetworkOutputs);
                else
                    [validationResultsFolds(:, currentChromosome), testResultsFolds(:, currentChromosome)] = multiCubeELMParallel(outputsNo, inputTrain, inputValidation, inputTest, outputTrain, outputValidation, outputTest, population(currentChromosome, :), transferFunction, testSet, datasetType, keepNetworkOutputs);
                end
            else
                if keepNetworkOutputs == 1
                    [validationResultsFolds(:, currentChromosome), testResultsFolds(:, currentChromosome), networkOutputsTable(:, currentChromosome)] = multiCubeELM(outputsNo, inputTrain, inputValidation, inputTest, outputTrain, outputValidation, outputTest, population(currentChromosome, :), transferFunction, testSet, datasetType, keepNetworkOutputs);
                else
                    [validationResultsFolds(:, currentChromosome), testResultsFolds(:, currentChromosome)] = multiCubeELM(outputsNo, inputTrain, inputValidation, inputTest, outputTrain, outputValidation, outputTest, population(currentChromosome, :), transferFunction, testSet, datasetType, keepNetworkOutputs);
                end
            end
        end
    end
end

fitness = mean(validationResultsFolds);
selectedPopulation = cell(populationSize, size(population, 2));
selectedValidationResultsFolds = zeros(foldsNo, populationSize);
if testSet == 1
    selectedTestResultsFolds = zeros(foldsNo, populationSize);
end

if datasetType == 1
    [~, bestChromosomesPositions]  = sort(fitness, 'descend');
else
    [~, bestChromosomesPositions]  = sort(fitness, 'ascend');
end

for currentChromosome = 1:populationSize
    selectedPopulation(currentChromosome, :) = population(bestChromosomesPositions(1, currentChromosome), :);
end

for currentChromosome = 1:populationSize
    selectedValidationResultsFolds(:, currentChromosome) = validationResultsFolds(:, bestChromosomesPositions(1, currentChromosome));
    if testSet == 1
        selectedTestResultsFolds(:, currentChromosome) = testResultsFolds(:, bestChromosomesPositions(1, currentChromosome));
    end
end

varargout{1} = selectedPopulation;
varargout{2} = selectedValidationResultsFolds;
if testSet == 1
    varargout{3} = selectedTestResultsFolds;
    if keepNetworkOutputs == 1
        varargout{4} = networkOutputsTable(:, bestChromosomesPositions(1, 1));
    end
end
end

