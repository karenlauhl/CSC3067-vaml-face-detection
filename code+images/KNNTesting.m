function prediction = KNNTesting(testImage, modelNN, K)
    % Extract training data and labels from the model
    trainingImages = modelNN.neighbours;
    trainingLabels = modelNN.labels;
    
    % Compute Euclidean distances from the test image to all training images
    distances = zeros(size(trainingImages, 1), 1);
    for i = 1:size(trainingImages, 1)
        distances(i) = EuclideanDistance(testImage, trainingImages(i, :));
    end
    
    % Sort distances to find the K closest neighbors
    [sortedDistances, sortedIndices] = sort(distances);
    closestLabels = trainingLabels(sortedIndices(1:K)); % Labels of K nearest neighbors
    
    % Use majority voting to decide the final class
    prediction = mode(closestLabels); % Mode returns the most frequent label
end
