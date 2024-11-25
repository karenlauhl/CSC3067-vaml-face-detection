function halfhalfsplit(trainFile, testFile)
    % halfhalfsplit: Combines existing train and test .cdataset files and re-splits them 50-50.
    % Parameters:
    %   trainFile - Path to the current training .cdataset file
    %   testFile - Path to the current testing .cdataset file

    % Generate output file names
    [trainPath, trainName, ~] = fileparts(trainFile);
    outputTrainFile = fullfile(trainPath, sprintf('halfhalf_%s_train.cdataset', trainName));
    outputTestFile = fullfile(trainPath, sprintf('halfhalf_%s_test.cdataset', trainName));
    
    % Read the train and test datasets
    [Header, trainSamples] = readCdataset(trainFile);
    [Header, testSamples] = readCdataset(testFile);
    
    % Combine the samples
    combinedSamples = [trainSamples; testSamples];
    
    % Shuffle the combined samples
    rng('default'); % For reproducibility
    numSamples = length(combinedSamples);
    indices = randperm(numSamples);
    
    % Split the data into 50-50
    numTrain = round(numSamples * 0.5);
    trainSamples = combinedSamples(indices(1:numTrain));
    testSamples = combinedSamples(indices(numTrain+1:end));
    
    % Write the training dataset
    writeCdataset(outputTrainFile, Header, trainSamples);
    
    % Write the testing dataset
    writeCdataset(outputTestFile, Header, testSamples);
    
    fprintf('Re-splitting completed.\nTraining set saved to %s.\nTesting set saved to %s.\n', outputTrainFile, outputTestFile);
end

function [header, samples] = readCdataset(filePath)
    % Helper function to read a .cdataset file
    fid = fopen(filePath, 'r');
    if fid == -1
        error('Could not open file: %s', filePath);
    end
    header = cell(3, 1);
    header{1} = fgetl(fid); % Dataset type
    header{2} = fgetl(fid); % Image size
    header{3} = fgetl(fid); % Number of samples (ignored here)
    samples = {};
    line = fgetl(fid);
    while ischar(line)
        samples{end+1, 1} = line; %#ok<AGROW>
        line = fgetl(fid);
    end
    fclose(fid);
end

function writeCdataset(filePath, header, samples)
    % Helper function to write a .cdataset file
    fid = fopen(filePath, 'w');
    if fid == -1
        error('Could not open file for writing: %s', filePath);
    end
    fprintf(fid, '%s\n%s\n%d\n', header{1}, header{2}, length(samples));
    fprintf(fid, '%s\n', samples{:});
    fclose(fid);
end
