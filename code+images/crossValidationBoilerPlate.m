%% Training
% Main script for cross-validation with multiple classifiers
addpath("SVM-KM")

% Load all data from all.cdataset
[images, labels] = loadFaceImages("all.cdataset");

[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(images);
model1NN = NNtraining(Xpca, labels);
modelKNN = KNNtraining(Xpca, labels);
modelSVM = SVMtraining(Xpca, labels); % change this to SVM, 1-NN, k-NN, RF, or NN
% modelRF = RFtraining(images, labels); 
% modelNN = NNtraining(images, labels);

% Number of folds for cross-validation
numFolds = 5;


% Shuffle the data
numSamples = size(images, 1);
indices = randperm(numSamples);

% Divide data into folds
foldSize = floor(numSamples / numFolds);
folds = cell(numFolds, 1);

for i = 1:numFolds
    startIdx = (i-1)*foldSize + 1;
    if i < numFolds
        endIdx = i*foldSize;
    else
        endIdx = numSamples; % Include remaining samples in the last fold
    end
    folds{i} = indices(startIdx:endIdx);
end

% Initialize result storage for errors
errors1NN = zeros(numFolds, 1);
errorsKNN = zeros(numFolds, 1);
errorsSVM = zeros(numFolds, 1);
% errorsRF = zeros(numFolds, 1);
% errorsNN = zeros(numFolds, 1);

% Overall Metrics
% Initialize variables for storing metrics
totalTP = 0; totalTN = 0; totalFP = 0; totalFN = 0;

% Define sliding window parameters
windowSize = [27, 18];
stepSize = 7;
scales = [1, 0.7]; % 0.9, 0.8, 0.7, 0.6
iouThreshold = 0.1;
imageFilePath = 'im1.jpg'; % Image for sliding window
groupImage = imread(imageFilePath);

fprintf('Performing %d-Fold Cross-Validation...\n', numFolds);

% Cross-validation loop
for fold = 1:numFolds
    % Get training and test indices
    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);
    testpca = (testIdx - meanX) * eigenVectors;
   

    trainIdx = setdiff(indices, testIdx);
    testIdx = folds{fold};

    % Split data into training and testing sets
    % Get test and train indices
    testIdx = folds{fold};
    trainIdx = setdiff(indices, testIdx);
    
    % Split the data
    XTrain = images(trainIdx, :);
    YTrain = labels(trainIdx);
    XTest = images(testIdx, :);
    YTest = labels(testIdx);
    
    % PCA transformation
    meanXTrain = mean(XTrain, 1);
    XTrainPCA = (XTrain - meanXTrain) * eigenVectors;
    XTestPCA = (XTest - meanXTrain) * eigenVectors; % Use train mean for test data

    % Train and evaluate 1-NN
    model1NN = fitcknn(XTrainPCA, YTrain, 'NumNeighbors', 1);
    predictions1NN = predict(model1NN, XTestPCA);
    errors1NN(fold) = mean(predictions1NN ~= YTest);
    
    % Train and evaluate k-NN
    modelKNN = fitcknn(XTrainPCA, YTrain, 'NumNeighbors', 3);
    predictionsKNN = predict(modelKNN, XTestPCA);
    errorsKNN(fold) = mean(predictionsKNN ~= YTest);
    
    % Train and evaluate SVM
    modelSVM = fitcsvm(XTrainPCA, YTrain);
    predictionsSVM = predict(modelSVM, XTestPCA);
    errorsSVM(fold) = mean(predictionsSVM ~= YTest);

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

% Calculate final cross-validation errors
avgError1NN = mean(errors1NN);
avgErrorKNN = mean(errorsKNN);
avgErrorSVM = mean(errorsSVM);

% Calculate overall metrics
PrecisionAvg = totalTP / (totalTP + totalFP);
RecallAvg = totalTP / (totalTP + totalFN);
SpecificityAvg = totalTN / (totalTN + totalFP);
F1Avg = 2 * (PrecisionAvg * RecallAvg) / (PrecisionAvg + RecallAvg);
FalseAlarmRateAvg = totalFP / (totalFP + totalTN);
AccuracyAvg = (totalTP + totalTN) / (totalTP + totalTN + totalFP + totalFN);

% Display final metrics
fprintf('Final Cross-Validation Metrics:\n');
fprintf('Average Accuracy: %.4f\n', AccuracyAvg);
fprintf('Precision: %.4f, Recall: %.4f, Specificity: %.4f\n', PrecisionAvg, RecallAvg, SpecificityAvg);
fprintf('F1-Measure: %.4f, False Alarm Rate: %.4f\n', F1Avg, FalseAlarmRateAvg);
fprintf('Average Cross-Validation Errors (1-NN): %.2f%%\n', avgError1NN * 100);
fprintf('Average Cross-Validation Errors (k-NN): %.2f%%\n', avgErrorKNN * 100);
fprintf('Average Cross-Validation Errors (SVM): %.2f%%\n', avgErrorSVM * 100);


