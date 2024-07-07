function [inputMatrix, outputMatrix] = createInputOutputSets(dataSet)
getHeaders = dataSet.Properties.VarNames;
if strcmp(getHeaders{1, 1}(1), 'x')
    [countInputs, countOutputs] = countInputsOutputs(getHeaders);
    inputMatrix = dataset2matrix(dataSet(:, 1:end - countOutputs));
    outputMatrix = dataset2matrix(dataSet(:, countInputs + 1:end));
else
    foldsNo = size(dataSet, 2);
    getHeaders = dataSet{1, 1}.Properties.VarNames;
    [countInputs, countOutputs] = countInputsOutputs(getHeaders);
    inputMatrix = cell(1, foldsNo);
    outputMatrix = cell(1, foldsNo);
    for currentFold = 1:foldsNo
        inputMatrix{1, currentFold} = dataset2matrix(dataSet{1, currentFold}(:, 1:end - countOutputs));
        outputMatrix{1, currentFold} = dataset2matrix(dataSet{1, currentFold}(:, countInputs + 1:end));
    end
end
end