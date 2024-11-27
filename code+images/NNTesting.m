% idid
function prediction = NNTesting(testImage, modelNN)
    % Extract training data and labels from the model
    trainingImages = modelNN.neighbours;
    trainingLabels = modelNN.labels;
    
    % Initialize variables to store the minimum distance and prediction
    minDistance = inf;
    prediction = -1;
    
    % Loop through all training samples
    for i = 1:size(trainingImages, 1)
        % Compute the Euclidean distance to the current training sample
        distance = EuclideanDistance(testImage, trainingImages(i, :));
        
        % Update minimum distance and prediction if necessary
        if distance < minDistance
            minDistance = distance;
            prediction = trainingLabels(i); % Assign the label of the closest training sample
        end
    end
end
