function prediction = NNTesting(testImage, modelNN)
    % Normalize the test image
    testImage = testImage ./ max(testImage(:));
    
    % Initialize minimum distance and prediction
    minDistance = inf;
    prediction = [];
    
    % Loop through each training sample
    for i = 1:size(modelNN.neighbours, 1)
        % Get the training sample and label
        trainingSample = modelNN.neighbours(i, :);
        trainingLabel = modelNN.labels(i);

        % Compute the distance (can replace with other metrics if required)
        dEuc = EuclideanDistance(testImage, trainingSample);

        % Update the prediction if a closer sample is found
        if dEuc < minDistance
            minDistance = dEuc;
            prediction = trainingLabel;
        end
    end
end
