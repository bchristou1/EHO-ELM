function poolSize = poolSizeCalc(neuronsNo, subCubeSizes)

poolSize = zeros(1, size(subCubeSizes, 2));

for currentSubCubeSize = 1:size(subCubeSizes, 2)
    if subCubeSizes(currentSubCubeSize) == 0
        subCubeSize = 1;
    else
        subCubeSize = subCubeSizes(currentSubCubeSize);
    end
    poolSize(1, currentSubCubeSize) =   3 * neuronsNo * (floor(neuronsNo / subCubeSize));
end

end
