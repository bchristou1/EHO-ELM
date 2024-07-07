close all;
clear;
clc;

% User defined parameters
inputDatasetName = 'datasetName'; % The input dataset name without the file extension
hiddenNodesNo = 10; % The number of hidden layer neurons
inputRange = [-1 1]; % The weight initialization range (e.g. [-1, 1]);
keepNetworkOutputs = 1; %Store Test Set Network Outputs? (0 for No and 1 for Yes)
mutationStep = 2; %Mutation step size
experimentsNo = 10; % The number of experiment runs
transferFunction = 'sig'; % transferFunction: The transfer function type - 'sig' for sigmoid,
%                                              - 'sin' for sinusoid
%                                              - 'tribas' for for triangular basis function
%                                              - 'radbas' for radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
%                                              - 'gaussian' for gaussian function
multiCoreCPU = 1; %Run in parallel? (0 for No and 1 for Yes)

% Automatically defined data
currentAllowedRun = 1;
totalAllowedRuns = ceil(5 / mutationStep); % The number of allowed runs
logicalCoresNo = feature('numCores'); % The number of logical cores

outputFolder = './results';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end
if  2 * hiddenNodesNo > 50
    populationSize = 50;
else
    populationSize =  2 * hiddenNodesNo;
end

if multiCoreCPU == 1
    %Start the parallel pool
    parallelPool = gcp('nocreate');
    if isempty(parallelPool)
        parallelPool = parpool('local', logicalCoresNo);
    end
end

datasetLocation = sprintf('dataset/%s', inputDatasetName); % The dataset location
load(datasetLocation); % Load the dataset
foldsNo = size(trainingSetCV, 2); % The number of folds
[inputsNo, outputsNo]  = findInputsOutputsNo(trainingSetCV{1, 1}.Properties.VarNames); % The number of neural network inputs and outputs
if inputsNo < 5
    subCubeSizes = sort([1:inputsNo]); % The vector containing the neuron sizes
else
    subCubeSizes = sort([1:5]); % The vector containing the neuron sizes
end

experimentsHeaders = constructCustomHeader(1:experimentsNo, 'experiment');
validationResults = dataset({cell(1, experimentsNo), experimentsHeaders{:}});
testResults = dataset({cell(1, experimentsNo), experimentsHeaders{:}});
validationResultsHeaders = {'GenerationsNo', 'ValidationResult', 'ValidationResultsFolds'};
if keepNetworkOutputs == 1
    testResultsHeaders = {'GenerationsNo', 'TestResult', 'TestResultsFolds', 'TestOutputsFolds'};
else
    testResultsHeaders = {'GenerationsNo', 'TestResult', 'TestResultsFolds'};
end
[inputTrain, outputTrain] = createInputOutputSets(trainingSetCV);
[inputValidation, outputValidation] = createInputOutputSets(validationSetCV);
[inputTest, outputTest] = createInputOutputSets(testSet);
entriesNo = size(trainingSetCV{1, 1}, 1) + size(validationSetCV{1, 1}, 1) + size(testSet, 1); % The number of entries
timeTable = dataset({zeros(1, experimentsNo), experimentsHeaders{:}});
if datasetType == 1
    outputsNo = max([max(outputTrain{1, 1}(:, 1)), max(outputValidation{1, 1}(:, 1)), max(outputTest(:, 1))]);
else
    outputsNo = size(outputTrain{1, 1}, 2);
end

for currentExperiment = 1:experimentsNo

    tStart = tic;

    validationResultsFolds = [];
    offspring = [];
    testSet = 0;

    %Create the initial population
    if multiCoreCPU == 1
        population = createInitialPopulationParallel(inputsNo, outputsNo, hiddenNodesNo, subCubeSizes, populationSize, inputRange);
    else
        population = createInitialPopulation(inputsNo, outputsNo, hiddenNodesNo, subCubeSizes, populationSize, inputRange);
    end
    %Start the evolution process
    currentGeneration = 0;
    while currentGeneration >= 0
        currentGeneration = currentGeneration + 1;
        if currentAllowedRun == totalAllowedRuns
            currentAllowedRun = 1;
            testSet = 1;
            if keepNetworkOutputs == 1
                [population, validationResultsFolds, testResultsFolds, testNetworkOutputs] = selection(validationResultsFolds, population, offspring, populationSize, outputsNo, inputTrain, inputValidation, inputTest, outputTrain, outputValidation, outputTest, transferFunction, testSet, datasetType, multiCoreCPU, keepNetworkOutputs);
            else
                [population, validationResultsFolds, testResultsFolds] = selection(validationResultsFolds, population, offspring, populationSize, outputsNo, inputTrain, inputValidation, inputTest, outputTrain, outputValidation, outputTest, transferFunction, testSet, datasetType, multiCoreCPU, keepNetworkOutputs);
            end
            if datasetType == 1
                fprintf('Experiment: %d/%d, Generation: %d, Validation Results: %.2f%%, Test Results: %.2f%%\n', currentExperiment, experimentsNo, currentGeneration, mean(validationResultsFolds(:, 1)) * 100, mean(testResultsFolds(:, 1)) * 100);
            else
                fprintf('Experiment: %d/%d, Generation: %d, Validation Results: %.6f, Test Results: %.6f\n', currentExperiment, experimentsNo, currentGeneration, mean(validationResultsFolds(:, 1)), mean(testResultsFolds(:, 1)));
            end
            validationResults{1, currentExperiment} = dataset({{currentGeneration, mean(validationResultsFolds(:, 1)), validationResultsFolds(:, 1)'}, validationResultsHeaders{:}});
            if keepNetworkOutputs == 1
                testResults{1, currentExperiment} = dataset({{currentGeneration, mean(testResultsFolds(:, 1)), testResultsFolds(:, 1)', testNetworkOutputs'}, testResultsHeaders{:}});
            else
                testResults{1, currentExperiment} = dataset({{currentGeneration, mean(testResultsFolds(:, 1)), testResultsFolds(:, 1)'}, testResultsHeaders{:}});
            end
            break;
        else
            [population, validationResultsFolds] = selection(validationResultsFolds, population, offspring, populationSize, outputsNo, inputTrain, inputValidation, inputTest, outputTrain, outputValidation, outputTest, transferFunction, testSet, datasetType, multiCoreCPU, keepNetworkOutputs);
            if datasetType == 1
                fprintf('Experiment: %d/%d, Generation: %d, Validation Results: %.2f%%\n', currentExperiment, experimentsNo, currentGeneration, mean(validationResultsFolds(:, 1)) * 100);
            else
                fprintf('Experiment: %d/%d, Generation: %d, Validation Results: %.6f\n', currentExperiment, experimentsNo, currentGeneration, mean(validationResultsFolds(:, 1)));
            end
        end

        %Set the counter
        if currentGeneration > 1
            previousBestFitness = currentBestFitness;
            count = count + mutationStep;
        end

        currentBestFitness = mean(validationResultsFolds(:, 1));

        if datasetType == 1
            if (currentGeneration == 1) || (currentGeneration > 1 && currentBestFitness > previousBestFitness)
                count = 1;
            end
        else
            if (currentGeneration == 1) || (currentGeneration > 1 && currentBestFitness < previousBestFitness)
                count = 1;
            end
        end

        if count <= 5
            mutationRate = count * 0.1;
            currentAllowedRun = 1;
        else
            mutationRate = 0.5;
            currentAllowedRun = currentAllowedRun + 1;
        end

        fprintf('Mutation Rate: %f.\n', mutationRate);

        offspring = crossover(hiddenNodesNo, population, subCubeSizes);
        population = mutation(population, offspring, mutationRate, hiddenNodesNo, inputRange);
    end
    fprintf('\n');

    timeTable{1, currentExperiment} = toc(tStart);

end
datasetHash = DataHash(sprintf('%s.mat',datasetLocation), 'file');
save(sprintf('%s/%s', outputFolder, inputDatasetName), 'validationResults', 'testResults', 'hiddenNodesNo', 'datasetType', 'entriesNo', 'inputsNo', 'outputsNo', 'inputDatasetName', 'datasetHash', 'timeTable', 'logicalCoresNo', 'multiCoreCPU', '-v7.3');

if multiCoreCPU == 1

    % Shut down the parallel pool
    parallelPool = gcp('nocreate');
    if ~isempty(parallelPool)
        delete(parallelPool);
    end
end

fprintf('Job Done.\n');

