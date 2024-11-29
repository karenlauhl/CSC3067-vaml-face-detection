% Main script for cross-validation with multiple classifiers

% Load all data from all.cdataset
[images, labels] = loadFaceImages("all.cdataset");

model1NN = NNtraining(images, labels);
modelKNN = KNNtraining(images, labels);
modelSVM = SVMtraining(images, labels); % change this to SVM, 1-NN, k-NN, RF, or NN
% modelRF = RFtraining(images, labels); 
% modelNN = NNtraining(images, labels);

% Number of folds for cross-validation
numFolds = 5;
cv = cvpartition(labels, 'KFold', numFolds);

% Initialize result storage for errors
errors1NN = zeros(cv.NumTestSets, 1);
errorsKNN = zeros(cv.NumTestSets, 1);
errorsSVM = zeros(cv.NumTestSets, 1);
% errorsRF = zeros(cv.NumTestSets, 1);
% errorsNN = zeros(cv.NumTestSets, 1);

% Define sliding window parameters
windowSize = [27, 18];
stepSize = 7;
scales = [1, 0.7]; % 0.9, 0.8, 0.7, 0.6
iouThreshold = 0.1;
imageFilePath = ['im1.jpg']; % Image for sliding window
groupImage = imread(imageFilePath);

fprintf('Performing %d-Fold Cross-Validation...\n', numFolds);

% Cross-validation loop
for fold = 1:cv.NumTestSets
    % Get training and test indices
    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);

    % Split data into training and testing sets
    XTrain = images(trainIdx, :);
    YTrain = labels(trainIdx);
    XTest = images(testIdx, :);
    YTest = labels(testIdx);

    
    % Train and evaluate 1-NN
    model1NN = fitcknn(XTrain, YTrain, 'NumNeighbors', 1);
    predictions1NN = predict(model1NN, XTest);
    errors1NN(fold) = mean(predictions1NN ~= YTest);
    
    % Train and evaluate k-NN (e.g., k = 3)
    modelKNN = fitcknn(XTrain, YTrain, 'NumNeighbors', 3);
    predictionsKNN = predict(modelKNN, XTest);
    errorsKNN(fold) = mean(predictionsKNN ~= YTest);
    
    % Train and evaluate SVM
    modelSVM = fitcsvm(XTrain, YTrain);
    predictionsSVM = predict(modelSVM, XTest);
    errorsSVM(fold) = mean(predictionsSVM ~= YTest);

    % % Train and evaluate Random Forest (Tree Ensemble)
    % modelRF = fitcensemble(XTrain, YTrain, 'Method', 'Bag', 'NumLearningCycles', 50);
    % predictionsRF = predict(modelRF, XTest);
    % errorsRF(fold) = mean(predictionsRF ~= YTest);
    % 
    % % Trainf and evaluate Neural Network
    % modelNN = fitNeuralNetwork(XTrain, YTrain);
    % predictionsNN = predictNeuralNetwork(modelNN, XTest);
    % errorsNN(fold) = mean(predictionsNN ~=YTest);

    % Run Sliding Window Detection using 1-NN as an example
    fprintf('Fold %d: Running sliding window detection with 1-NN...\n', fold);
    [detections, outputImage] = crossValidationFullImageSlidingWindowNMS(groupImage, model1NN, ...
        windowSize, stepSize, scales, iouThreshold);

    % Run sliding window detection using k-NN as an example
    fprintf('Fold %d: Running sliding window detection with k-NN...\n', fold);
    [detections, outputImage] = crossValidationFullImageSlidingWindowNMS(groupImage, modelKNN, ...
        windowSize, stepSize, scales, iouThreshold);

    % Run sliding window detection using SVM as an example
    fprintf('Fold %d: Running sliding window detection with SVM...\n', fold);
    [detections, outputImage] = crossValidationFullImageSlidingWindowNMS(groupImage, modelSVM, ...
        windowSize, stepSize, scales, iouThreshold);

    % % % Run sliding window detection using Neural Network as an example
    % fprintf('Fold %d: Running sliding window detection with Neural Network...\n', fold);
    % [detections, outputImage] = crossValidationFullImageSlidingWindowNMS(groupImage, modelNN, ...
    %     windowSize, stepSize, scales, iouThreshold);
end

% Calculate average cross-validation errors
avgError1NN = mean(errors1NN);
avgErrorKNN = mean(errorsKNN);
avgErrorSVM = mean(errorsSVM);
% avgErrorRF = mean(errorsRF);
% avgErrorNN = mean(errorsNN);

fprintf('Average Cross-Validation Error (1-NN): %.2f%%\n', avgError1NN * 100);
fprintf('Average Cross-Validation Error (k-NN): %.2f%%\n', avgErrorKNN * 100);
fprintf('Average Cross-Validation Error (SVM): %.2f%%\n', avgErrorSVM * 100);
% fprintf('Average Cross-Validation Error (Random Forest): %.2f%%\n', avgErrorRF * 100);
% fprintf('Average Cross-Validation Error (NN): %.2f%%\n', avgErrorNN * 100);
