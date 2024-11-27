% Main workflow script for face detection
%% Training

% Load training and test data
% there is total 670= 45*4 + 49*10 in train as calculated
[trainImages, trainLabels] = loadFaceImages('face_train.cdataset');   
[testImages, testLabels] = loadFaceImages('face_test.cdataset');

allImages=[trainImages; testImages];  % , Concatenation horin and ; does it vertical.
allLabels=[trainLabels; testLabels];  % ; does it vertical.

% Train the classifier
model = trainModel(trainImages, trainLabels);

%% Testing and Evaluation
% Evaluate the model on the test set
[labels, classificationResult] = evaluateModel(model, testImages, testLabels);
comparison = (labels==classificationResult);
Accuracy = sum(comparison)/length(comparison)

% Type 1 and Type 2 error
TP= sum ((labels==classificationResult) & (labels==1))
TN= sum ((labels==classificationResult) & (labels==-1))
FP= sum ((labels~=classificationResult) & (labels==-1))
FN= sum ((labels~=classificationResult) & (labels==1))

% Metrics
Precision = TP / (TP+FP)
Recall = TP / (TP+FN)
Specificity = TN / (TN+FP)
Sensitivity = TP / (TP+FN)
Fmeasure= 2 * TP / (2 * TP+FN+FP)
False_alarm_rate = 1 - Specificity

%% Detection
% Load the image file
imageFilePath = 'im3.jpg'; % Replace with your actual image file path
groupImage = imread(imageFilePath);

% Define parameters
windowSize = [27, 18];
stepSize = 7;
scales = [1,0.7]; % Example scale levels for the image pyramid   ,0.9, 0.8, 0.7, 0.6
iouThreshold=0.1;

%Run sliding window face detection
finalDetections = slidingWindowFaceDetection(groupImage, model, windowSize, stepSize, scales,iouThreshold);






% %Without using the gabor feature is 60%
% % Load training data
% [trainImages, trainLabels] = loadFaceImages('face_train.cdataset',1);
% 
% % Train a KNN classifier
% Mdl = fitcknn(trainImages, trainLabels, 'NumNeighbors', 3);
% 
% % Load test data
% [testImages, testLabels] = loadFaceImages('face_test.cdataset',1);
% 
% % Predict using the trained KNN model
% predictedLabels = predict(Mdl, testImages);
% 
% % Evaluate accuracy
% accuracy = sum(predictedLabels == testLabels) / numel(testLabels);
% fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);