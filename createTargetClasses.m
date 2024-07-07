function [outputTable, classesNo] = createTargetClasses(inputTable, classesNo, varargin)
if isempty(varargin)
    varargin{1} = -1;
end
samplesNo = length(inputTable);
outputTable(1:samplesNo, 1:classesNo) = varargin{1};
for i = 1:samplesNo
    outputTable(i, inputTable(i)) = 1;
end

