% Load training and test data
% there is total 670= 45*4 + 49*10 in train as calculated
[trainImages, trainLabels] = loadFaceImages('face_train.cdataset');   
[testImages, testLabels] = loadFaceImages('face_test.cdataset');

%allImages=[trainImages; testImages];  % , Concatenation horin and ; does it vertical.
%allLabels=[trainLabels; testLabels];  % ; does it vertical.

% %% NN

% %Nntesting -------------------------
% modelNN = NNtraining(trainImages, trainLabels);

% for i=1:size(testImages,1)
%     testnumber= testImages(i,:);
%     classificationResult(i,1) = NNTesting(testnumber, modelNN);
% end 

% % Evaluate NN accuracy
% accuracy_nn = sum(testLabels==classificationResult) / numel(testLabels);
% fprintf('Test Accuracy NN: %.2f%%\n', accuracy_nn * 100);

% %% KNN

% %KNntesting -------------------------
% modelKNN = KNNtraining(trainImages, trainLabels);

% for i=1:size(testImages,1)
%     testnumber= testImages(i,:);
%     classificationResult(i,1) = KNNTesting(testnumber, modelKNN,11);
% end 

% % Evaluate K-NN accuracy
% accuracyKnn = sum(testLabels==classificationResult) / numel(testLabels);
% fprintf('Test Accuracy KNN: %.2f%%\n', accuracyKnn * 100);

%% SVM Testing
modelSVM = SVMtraining(trainImages, trainLabels);

for i=1:size(testImages,1)
    testnumber = testImages(i,:);
    classificationResult(i,1) = SVMTesting(testnumber, modelSVM);
end

% Evaluate SVM accuracy
accuracySVM = sum(testLabels == classificationResult) / numel(testLabels);
fprintf('Test Accuracy SVM: %.2f%%\n', accuracySVM * 100);








% % Train the classifier
% model = trainModel(trainImages, trainLabels);
% 
% % Evaluate the model on the test set
% accuracy = evaluateModel(model, testImages, testLabels);
% 
% 
% % Load the image file
% imageFilePath = 'im3.jpg'; % Replace with your actual image file path
% groupImage = imread(imageFilePath);
% 
% % Define parameters
% windowSize = [27, 18];
% stepSize = 7;
% scales = [1,0.7]; % Example scale levels for the image pyramid   ,0.9, 0.8, 0.7, 0.6
% iouThreshold=0.1;
% 
% %Run sliding window face detection
% finalDetections = slidingWindowFaceDetection(groupImage, model, windowSize, stepSize, scales,iouThreshold);
% 





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