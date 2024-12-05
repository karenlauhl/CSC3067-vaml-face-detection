% Clear workspace
clear all;
close all;

addpath("SVM-KM\")

% Load training and test data
[trainImages, trainLabels] = loadFaceImages('face_train.cdataset');   
[testImages, testLabels] = loadFaceImages('face_test.cdataset');

% % Load halfhalf training and test data
% [trainImages, trainLabels] = loadFaceImages('halfhalf_face_train.cdataset');   
% [testImages, testLabels] = loadFaceImages('halfhalf_face_test.cdataset');

% Normalize image data
trainImages = double(trainImages) / 255;
testImages = double(testImages) / 255;

%% Training
% Train a 1NN model directly using training images
full_modelNN = NNtraining(trainImages, trainLabels);
% Train a KNN model directly using training images
full_modelKNN = KNNtraining(trainImages, trainLabels);
% Train a SVM model directly using training images
full_modelSVM = SVMtraining(trainImages, trainLabels);

%% 1NN Testing
for i=1:size(testImages,1)
    testnumber= testImages(i, :);
    classificationResult(i,1) = NNTesting(testnumber, full_modelNN);
end

if length(testLabels) ~= length(classificationResult)
    error('testLabels and classificationResult must have the same length.');
end

%% 1NN Evaluation
disp('Evaluating NN Model:')

% Finally we compared the predicted classification from our mahcine
% learning algorithm against the real labelling of the testing image
comparison = (testLabels==classificationResult);

%Accuracy is the most common metric. It is defiend as the numebr of
%correctly classified samples/ the total number of tested samples
Accuracy = sum(comparison)/length(comparison)

% Evaluate accuracy
% Type 1 and Type 2 error
TP= sum ((testLabels==classificationResult) & (testLabels==1))
TN= sum ((testLabels==classificationResult) & (testLabels==-1))
FP= sum ((testLabels~=classificationResult) & (testLabels==-1))
FN= sum ((testLabels~=classificationResult) & (testLabels==1))

% Displaying values in a bar chart
eval = [TP, TN, FP, FN];
evalNN = figure;
bar(eval);
set(gca, 'XTickLabel', {'TP', 'TN', 'FP', 'FN'});
xlabel('Classification');
ylabel('Value');
title('Type 1 and Type 2 Error for NN');
grid on;

% Metrics
Precision = TP / (TP+FP)
Recall = TP / (TP+FN)
Specificity = TN / (TN+FP)
Sensitivity = TP / (TP+FN)
Fmeasure= 2 * TP / (2 * TP+FN+FP)
False_alarm_rate = 1 - Specificity

% Displaying values in a bar chart
eval = [Precision, Recall, Specificity, Sensitivity, Fmeasure, False_alarm_rate];
metricsNN = figure;
bar(eval);
set(gca, 'XTickLabel', {'Precision', 'Recall', 'Specificity', 'Sensitivity', 'Fmeasure', 'False Alarm Rate'});
xlabel('Classification');
ylabel('Value');
title('Metrics for NN');
grid on;
hold off

%% KNN Testing
for i=1:size(testImages,1)
    testnumber= testImages(i, :);
    classificationResult(i,1) = KNNTesting(testnumber, full_modelKNN, 5);
end

if length(testLabels) ~= length(classificationResult)
    error('testLabels and classificationResult must have the same length.');
end

%% KNN Evaluation
disp('Evaluating KNN Model:')

% Finally we compared the predicted classification from our mahcine
% learning algorithm against the real labelling of the testing image
comparison = (testLabels==classificationResult);

%Accuracy is the most common metric. It is defiend as the numebr of
%correctly classified samples/ the total number of tested samples
Accuracy = sum(comparison)/length(comparison)

% Evaluate accuracy
% Type 1 and Type 2 error
TP= sum ((testLabels==classificationResult) & (testLabels==1))
TN= sum ((testLabels==classificationResult) & (testLabels==-1))
FP= sum ((testLabels~=classificationResult) & (testLabels==-1))
FN= sum ((testLabels~=classificationResult) & (testLabels==1))

% Displaying values in a bar chart
eval = [TP, TN, FP, FN];
evalKNN = figure;
bar(eval);
set(gca, 'XTickLabel', {'TP', 'TN', 'FP', 'FN'});
xlabel('Classification');
ylabel('Value');
title('Type 1 and Type 2 Error for KNN');
grid on;

% Metrics
Precision = TP / (TP+FP)
Recall = TP / (TP+FN)
Specificity = TN / (TN+FP)
Sensitivity = TP / (TP+FN)
Fmeasure= 2 * TP / (2 * TP+FN+FP)
False_alarm_rate = 1 - Specificity

% Displaying values in a bar chart
eval = [Precision, Recall, Specificity, Sensitivity, Fmeasure, False_alarm_rate];
metricsKNN = figure;
bar(eval);
set(gca, 'XTickLabel', {'Precision', 'Recall', 'Specificity', 'Sensitivity', 'Fmeasure', 'False Alarm Rate'});
xlabel('Classification');
ylabel('Value');
title('Metrics for KNN');
grid on;
hold off

%% SVM Testing
for i=1:size(testImages,1)
    testnumber= testImages(i, :);
    classificationResult(i,1) = SVMTesting(testnumber, full_modelSVM);
end

if length(testLabels) ~= length(classificationResult)
    error('testLabels and classificationResult must have the same length.');
end

%% SVM Evaluation
disp('Evaluating SVM Model:')

% Finally we compared the predicted classification from our mahcine
% learning algorithm against the real labelling of the testing image
comparison = (testLabels==classificationResult);

%Accuracy is the most common metric. It is defiend as the numebr of
%correctly classified samples/ the total number of tested samples
Accuracy = sum(comparison)/length(comparison)

% Evaluate accuracy
% Type 1 and Type 2 error
TP= sum ((testLabels==classificationResult) & (testLabels==1))
TN= sum ((testLabels==classificationResult) & (testLabels==-1))
FP= sum ((testLabels~=classificationResult) & (testLabels==-1))
FN= sum ((testLabels~=classificationResult) & (testLabels==1))

% Displaying values in a bar chart
eval = [TP, TN, FP, FN];
evalSVM = figure;
bar(eval);
set(gca, 'XTickLabel', {'TP', 'TN', 'FP', 'FN'});
xlabel('Classification');
ylabel('Value');
title('Type 1 and Type 2 Error for SVM');
grid on;

% Metrics
Precision = TP / (TP+FP)
Recall = TP / (TP+FN)
Specificity = TN / (TN+FP)
Sensitivity = TP / (TP+FN)
Fmeasure= 2 * TP / (2 * TP+FN+FP)
False_alarm_rate = 1 - Specificity

% Displaying values in a bar chart
eval = [Precision, Recall, Specificity, Sensitivity, Fmeasure, False_alarm_rate];
metricsSVM = figure;
bar(eval);
set(gca, 'XTickLabel', {'Precision', 'Recall', 'Specificity', 'Sensitivity', 'Fmeasure', 'False Alarm Rate'});
xlabel('Classification');
ylabel('Value');
title('Metrics for SVM');
grid on;
hold off

%% Sliding Window Parameters
windowSize = [27, 18];
stepSize = 7;
scales = [1, 0.9, 0.8, 0.7];
iouThreshold = 0.3;
% Define the test images for sliding window detection
groupImage = {'im1.jpg', 'im2.jpg', 'im3.jpg', 'im4.jpg'};

% Run sliding window face detection with the 1NN trained model
finalDetections_1NN = slidingWindow_full_nn(groupImage, full_modelNN);
% Run sliding window face detection with the KNN trained model
finalDetections_KNN = slidingWindow_full_knn(groupImage, full_modelKNN);
% Run sliding window face detection with the SVM trained model
finalDetections_SVM = slidingWindow_full_Svm(groupImage, full_modelSVM);

%% Display results
disp('Sliding window detection completed.');
