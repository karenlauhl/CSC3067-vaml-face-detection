% Main script for cross-validation with multiple classifiers
addpath("SVM-KM\")

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

%% Overall Metrics
% Initialize variables for storing metrics
totalTP = 0; totalTN = 0; totalFP = 0; totalFN = 0;

for fold = 1:cv.NumTestSets
    % Get training and test indices
    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);

    % Split data into training and testing sets
    XTrain = images(trainIdx, :);
    YTrain = labels(trainIdx);
    XTest = images(testIdx, :);
    YTest = labels(testIdx);

    % Train and evaluate SVM (as an example)
    modelSVM = fitcsvm(XTrain, YTrain);
    predictionsSVM = predict(modelSVM, XTest);

    % Confusion Matrix
    [confMatrix, ~] = confusionmat(YTest, predictionsSVM);

    % Extract TP, TN, FP, FN from the confusion matrix
    TP = confMatrix(2, 2);
    TN = confMatrix(1, 1);
    FP = confMatrix(1, 2);
    FN = confMatrix(2, 1);

    % Accumulate values for averaging later
    totalTP = totalTP + TP;
    totalTN = totalTN + TN;
    totalFP = totalFP + FP;
    totalFN = totalFN + FN;

    % Compute fold-level metrics
    Precision = TP / (TP + FP);
    Recall = TP / (TP + FN); % Same as Sensitivity
    Specificity = TN / (TN + FP);
    F1 = 2 * (Precision * Recall) / (Precision + Recall);
    FalseAlarmRate = FP / (FP + TN);
    Accuracy = (TP + TN) / (TP + TN + FP + FN);

    % Display fold metrics
    fprintf('Fold %d Metrics:\n', fold);
    fprintf('Precision: %.4f, Recall: %.4f, Specificity: %.4f\n', Precision, Recall, Specificity);
    fprintf('F1-Measure: %.4f, False Alarm Rate: %.4f, Accuracy: %.4f\n', F1, FalseAlarmRate, Accuracy);
end

% Calculate overall metrics
PrecisionAvg = totalTP / (totalTP + totalFP);
RecallAvg = totalTP / (totalTP + totalFN);
SpecificityAvg = totalTN / (totalTN + totalFP);
F1Avg = 2 * (PrecisionAvg * RecallAvg) / (PrecisionAvg + RecallAvg);
FalseAlarmRateAvg = totalFP / (totalFP + totalTN);
AccuracyAvg = (totalTP + totalTN) / (totalTP + totalTN + totalFP + totalFN);

% Display overall metrics
fprintf('Average Metrics:\n');
fprintf('Precision: %.4f, Recall: %.4f, Specificity: %.4f\n', PrecisionAvg, RecallAvg, SpecificityAvg);
fprintf('F1-Measure: %.4f, False Alarm Rate: %.4f, Accuracy: %.4f\n', F1Avg, FalseAlarmRateAvg, AccuracyAvg);
