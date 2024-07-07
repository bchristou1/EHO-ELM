function [countInputs, countOutputs] = countInputs(getHeaders)
countInputs = 0;
countOutputs = 0;
for currentEntry = 1:size(getHeaders, 2)
    if strcmp(getHeaders{1, currentEntry}(1), 'x')
        countInputs = countInputs + 1;
    else
        countOutputs = countOutputs + 1;
    end
end
end

