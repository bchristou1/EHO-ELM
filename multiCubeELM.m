function varargout = multiCubeELM(outputsNo, inputTrain, inputValidation, inputTest, outputTrain, outputValidation, outputTest, chromosome, transferFunction, testSet, datasetType, keepNetworkOutputs)

% inputTrain: The input training set
% inputValidation: The input validation set
% inputTest: The input test set
% outputTrain: The output training set
% outputValidation: The output validation set
% outputTest: The output test set
% chromosome: The chromosome
% transferFunction: The transfer function type - 'sig' for sigmoid,
%                                              - 'sin' for sinusoid
%                                              - 'tribas' for for triangular basis function
%                                              - 'radbas' for radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
%                                              - 'gaussian' for gaussian function
% problemType: The problem type - 0 for regression
%                               - 1 for classification

hiddenNodesNo = size(chromosome, 2) - outputsNo;
foldsNo = size(inputValidation, 2);
resultsValidationFolds = zeros(1, foldsNo);
networkOutputValuesTest = cell(1, foldsNo);

if testSet ~= 0
    resultsTestFolds = zeros(1, foldsNo);
    samplesNoTest = size(inputTest, 1);
    HTest = zeros(samplesNoTest, hiddenNodesNo);
    
    % Calculate the Sigma-Pi hidden layer matrix H for the test set
    for currentSample = 1:samplesNoTest
        for currentHiddenNodeNo = 1:hiddenNodesNo
            if strcmp(transferFunction, 'sig') || strcmp(transferFunction, 'sin') || strcmp(transferFunction, 'tribas') || strcmp(transferFunction, 'radbas') || strcmp(transferFunction, 'gaussian')
                if strcmp(transferFunction, 'sig')
                    HTest(currentSample, currentHiddenNodeNo) = logsig(multiCubeNeuron(inputTest(currentSample, :), cell2mat(chromosome{1, currentHiddenNodeNo}.hiddenLayerWeights), chromosome{1, currentHiddenNodeNo}.hiddenLayerNeuronStructure));
                end
                if strcmp(transferFunction, 'sin')
                    HTest(currentSample, currentHiddenNodeNo) = sin(multiCubeNeuron(inputTest(currentSample, :), cell2mat(chromosome{1, currentHiddenNodeNo}.hiddenLayerWeights), chromosome{1, currentHiddenNodeNo}.hiddenLayerNeuronStructure));
                end
                if strcmp(transferFunction, 'tribas')
                    HTest(currentSample, currentHiddenNodeNo) = tribas(multiCubeNeuron(inputTest(currentSample, :), cell2mat(chromosome{1, currentHiddenNodeNo}.hiddenLayerWeights), chromosome{1, currentHiddenNodeNo}.hiddenLayerNeuronStructure));
                end
                if strcmp(transferFunction, 'radbas')
                    HTest(currentSample, currentHiddenNodeNo) = radbas(multiCubeNeuron(inputTest(currentSample, :), cell2mat(chromosome{1, currentHiddenNodeNo}.hiddenLayerWeights), chromosome{1, currentHiddenNodeNo}.hiddenLayerNeuronStructure));
                end
                if strcmp(transferFunction, 'gaussian')
                    HTest(currentSample, currentHiddenNodeNo) = gaussian(multiCubeNeuron(inputTest(currentSample, :), cell2mat(chromosome{1, currentHiddenNodeNo}.hiddenLayerWeights), chromosome{1, currentHiddenNodeNo}.hiddenLayerNeuronStructure));
                end
            else
                error('Invalid transfer function type');
            end
        end
    end
end

for currentFold = 1:foldsNo
    samplesNoTrain = size(inputTrain{1, currentFold}, 1);
    samplesNoValidation = size(inputValidation{1, currentFold}, 1);
    
    % Contruct the classification outputs matrices
    if datasetType == 1
        outputTrain{1,currentFold} = createTargetClasses(outputTrain{1,currentFold}, outputsNo);
    end
    
    % Make the necessary initializations
    outputWeightsNo = zeros(1, outputsNo);
    for currentOutput = 1:outputsNo
        outputWeightsNo(1, currentOutput) = sum(2 .^ chromosome{1, currentOutput + hiddenNodesNo}.outputLayerNeuronStructure);
    end
    HTrain = zeros(samplesNoTrain, hiddenNodesNo);
    HValidation = zeros(samplesNoValidation, hiddenNodesNo);
    augSigmaPiHTrain = cell(1, outputsNo);
    augSigmaPiHValidation = cell(1, outputsNo);
    if testSet ~= 0
        augSigmaPiHTest = cell(1, outputsNo);
    end
    w = cell(1, outputsNo);
    for currentOutput = 1:outputsNo
        augSigmaPiHTrain{1, currentOutput} = zeros(samplesNoTrain, outputWeightsNo(1, currentOutput));
        augSigmaPiHValidation{1, currentOutput} = zeros(samplesNoValidation, outputWeightsNo(1, currentOutput));
        if testSet ~= 0
            augSigmaPiHTest{1, currentOutput} = zeros(samplesNoTest, outputWeightsNo(1, currentOutput));
        end
        w{1, currentOutput} = ones(1, outputWeightsNo(1, currentOutput));
    end
    
    % Calculate the Sigma-Pi hidden layer matrix H for the training set
    for currentSample = 1:samplesNoTrain
        for currentHiddenNodeNo = 1:hiddenNodesNo
            if strcmp(transferFunction, 'sig') || strcmp(transferFunction, 'sin') || strcmp(transferFunction, 'tribas') || strcmp(transferFunction, 'radbas') || strcmp(transferFunction, 'gaussian')
                if strcmp(transferFunction, 'sig')
                    HTrain(currentSample, currentHiddenNodeNo) = logsig(multiCubeNeuron(inputTrain{1, currentFold}(currentSample, :), cell2mat(chromosome{1, currentHiddenNodeNo}.hiddenLayerWeights), chromosome{1, currentHiddenNodeNo}.hiddenLayerNeuronStructure));
                end
                if strcmp(transferFunction, 'sin')
                    HTrain(currentSample, currentHiddenNodeNo) = sin(multiCubeNeuron(inputTrain{1, currentFold}(currentSample, :), cell2mat(chromosome{1, currentHiddenNodeNo}.hiddenLayerWeights), chromosome{1, currentHiddenNodeNo}.hiddenLayerNeuronStructure));
                end
                if strcmp(transferFunction, 'tribas')
                    HTrain(currentSample, currentHiddenNodeNo) = tribas(multiCubeNeuron(inputTrain{1, currentFold}(currentSample, :), cell2mat(chromosome{1, currentHiddenNodeNo}.hiddenLayerWeights), chromosome{1, currentHiddenNodeNo}.hiddenLayerNeuronStructure));
                end
                if strcmp(transferFunction, 'radbas')
                    HTrain(currentSample, currentHiddenNodeNo) = radbas(multiCubeNeuron(inputTrain{1, currentFold}(currentSample, :), cell2mat(chromosome{1, currentHiddenNodeNo}.hiddenLayerWeights), chromosome{1, currentHiddenNodeNo}.hiddenLayerNeuronStructure));
                end
                if strcmp(transferFunction, 'gaussian')
                    HTrain(currentSample, currentHiddenNodeNo) = gaussian(multiCubeNeuron(inputTrain{1, currentFold}(currentSample, :), cell2mat(chromosome{1, currentHiddenNodeNo}.hiddenLayerWeights), chromosome{1, currentHiddenNodeNo}.hiddenLayerNeuronStructure));
                end
            else
                error('Invalid transfer function type');
            end
        end
    end
    
    % Calculate the Sigma-Pi hidden layer matrix H for the validation set
    for currentSample = 1:samplesNoValidation
        for currentHiddenNodeNo = 1:hiddenNodesNo
            if strcmp(transferFunction, 'sig')
                HValidation(currentSample, currentHiddenNodeNo) = logsig(multiCubeNeuron(inputValidation{1, currentFold}(currentSample, :), cell2mat(chromosome{1, currentHiddenNodeNo}.hiddenLayerWeights), chromosome{1, currentHiddenNodeNo}.hiddenLayerNeuronStructure));
            end
            if strcmp(transferFunction, 'sin')
                HValidation(currentSample, currentHiddenNodeNo) = sin(multiCubeNeuron(inputValidation{1, currentFold}(currentSample, :), cell2mat(chromosome{1, currentHiddenNodeNo}.hiddenLayerWeights), chromosome{1, currentHiddenNodeNo}.hiddenLayerNeuronStructure));
            end
            if strcmp(transferFunction, 'tribas')
                HValidation(currentSample, currentHiddenNodeNo) = tribas(multiCubeNeuron(inputValidation{1, currentFold}(currentSample, :), cell2mat(chromosome{1, currentHiddenNodeNo}.hiddenLayerWeights), chromosome{1, currentHiddenNodeNo}.hiddenLayerNeuronStructure));
            end
            if strcmp(transferFunction, 'radbas')
                HValidation(currentSample, currentHiddenNodeNo) = radbas(multiCubeNeuron(inputValidation{1, currentFold}(currentSample, :), cell2mat(chromosome{1, currentHiddenNodeNo}.hiddenLayerWeights), chromosome{1, currentHiddenNodeNo}.hiddenLayerNeuronStructure));
            end
            if strcmp(transferFunction, 'gaussian')
                HValidation(currentSample, currentHiddenNodeNo) = gaussian(multiCubeNeuron(inputValidation{1, currentFold}(currentSample, :), cell2mat(chromosome{1, currentHiddenNodeNo}.hiddenLayerWeights), chromosome{1, currentHiddenNodeNo}.hiddenLayerNeuronStructure));
            end
        end
    end
    
    % Create the augmented matrix for the training set
    for currentSample = 1:samplesNoTrain
        for currentOutput = 1:outputsNo
            [~, augSigmaPiHTrain{1, currentOutput}(currentSample, :)] = multiCubeNeuron(HTrain(currentSample, :), w{1, currentOutput}, chromosome{1, currentOutput + hiddenNodesNo}.outputLayerNeuronStructure);
        end
    end
    
    % Create the augmented matrix for the validation set
    for currentSample = 1:samplesNoValidation
        for currentOutput = 1:outputsNo
            [~, augSigmaPiHValidation{1, currentOutput}(currentSample, :)] = multiCubeNeuron(HValidation(currentSample, :), w{1, currentOutput}, chromosome{1, currentOutput + hiddenNodesNo}.outputLayerNeuronStructure);
        end
    end
    
    if testSet ~= 0
        % Create the augmented matrix for the test set
        for currentSample = 1:samplesNoTest
            for currentOutput = 1:outputsNo
                [~, augSigmaPiHTest{1, currentOutput}(currentSample, :)] = multiCubeNeuron(HTest(currentSample, :), w{1, currentOutput}, chromosome{1, currentOutput + hiddenNodesNo}.outputLayerNeuronStructure);
            end
        end
    end
    
    % Calculate the output neuron weights
    currentFoldOutputWeights = cell(1, outputsNo);
    for currentOutput = 1:outputsNo
        currentFoldOutputWeights{1, currentOutput} = pinv(augSigmaPiHTrain{1, currentOutput}) * outputTrain{1,currentFold}(:, currentOutput);
    end
    
    % Calculate the validation set network output
    networkOutputValidation = zeros(samplesNoValidation, outputsNo);
    for currentOutput = 1:outputsNo
        networkOutputValidation(:, currentOutput) = augSigmaPiHValidation{1, currentOutput} * currentFoldOutputWeights{1, currentOutput};
    end
    
    if testSet ~= 0
        % Calculate the test set network output
        networkOutputTest = zeros(samplesNoTest, outputsNo);
        for currentOutput = 1:outputsNo
            networkOutputTest(:, currentOutput) = augSigmaPiHTest{1, currentOutput} * currentFoldOutputWeights{1, currentOutput};
        end
    end
    
    % Calculate the network accuracy
    if datasetType == 0
        resultsValidationFolds(1, currentFold) = mean(mean((outputValidation{1, currentFold} - networkOutputValidation) .^ 2));
        if testSet == 1
            resultsTestFolds(1, currentFold) = mean(mean((outputTest - networkOutputTest) .^ 2));
            if keepNetworkOutputs == 1
                networkOutputValuesTest{1, currentFold} = networkOutputTest;
            end
        end
    else
        if datasetType == 1
            errorsNoValidation = 0;
            for currentSample = 1:samplesNoValidation
                [~, currentClass] = max(networkOutputValidation(currentSample, :));
                if currentClass ~= outputValidation{1, currentFold}(currentSample)
                    errorsNoValidation =  errorsNoValidation + 1;
                end
            end
            resultsValidationFolds(1, currentFold) = 1 - (errorsNoValidation / samplesNoValidation);
            
            if testSet ~= 0
                errorsNoTest = 0;
                for currentSample = 1:samplesNoTest
                    [~, currentClass] = max(networkOutputTest(currentSample, :));
                    if currentClass ~= outputTest(currentSample)
                        errorsNoTest =  errorsNoTest + 1;
                    end
                end
                resultsTestFolds(1, currentFold) = 1 - (errorsNoTest / samplesNoTest);
                
                if keepNetworkOutputs == 1
                    networkOutputValuesTest{1, currentFold} = networkOutputTest;
                end
            end
        else
            error('Invalid problem type');
        end
    end
end

varargout{1} = resultsValidationFolds;
if testSet ~= 0
    varargout{2} = resultsTestFolds;
    if keepNetworkOutputs == 1
        varargout{3} = networkOutputValuesTest;
    end
end

end


