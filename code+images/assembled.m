% Main workflow script for face detection
%% Training

% Load training and test data
% there is total 670= 45*4 + 49*10 in train as calculated
[trainImages, trainLabels] = loadFaceImages('face_train.cdataset');   
[testImages, testLabels] = loadFaceImages('face_test.cdataset');

% Initialize matrix to store HOG features for all training images
numImages = size(trainImages, 1); % Number of images in the dataset
hogtrainMatrix = zeros(numImages, length(hog_feature_vector(reshape(trainImages(1, :), 27, 18)))); % Preallocate
% Loop through each image in the training set
for i = 1:numImages
    % Reshape each image from vector to 2D matrix
    imgt = reshape(trainImages(i, :), [27, 18]);
    % Extract HOG features for the current image
    hogtrainMatrix(i, :) = hog_feature_vector(imgt); % Store HOG features in the matrix
end

% Check the size of the HOG matrix (each row corresponds to a feature vector for one image)
disp(size(hogtrainMatrix));

% Initialize matrix for HOG features of test images
numTestImages = size(testImages, 1);
testHogMatrix = zeros(numTestImages, size(hogtrainMatrix, 2)); % Preallocate
% Loop through the test images and extract HOG features
for i = 1:numTestImages
    % Reshape the test image to 2D
    testImg = reshape(testImages(i, :), [27, 18]);
    % Extract HOG features for the test image
    testHogMatrix(i, :) = hog_feature_vector(testImg); % Store HOG features
end
disp(size(testHogMatrix));

%allImages=[trainImages; testImages];  % , Concatenation horin and ; does it vertical.
%allLabels=[trainLabels; testLabels];  % ; does it vertical.


%% NN Model Training
[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(hogtrainMatrix);
modelNN = NNtraining(Xpca, trainLabels);
%modelNN = NNtraining(hogtrainMatrix, trainLabels);

%% NN Testing

for i=1:size(testImages,1)
    testnumber= testHogMatrix(i,:);
    testpca = (testnumber - meanX) * eigenVectors;
    classificationResult(i,1) = NNTesting(testpca, modelNN);
end

%% Evaluation

% Finally we compared the predicted classification from our mahcine
% learning algorithm against the real labelling of the testing image
comparison = (testLabels==classificationResult);

%Accuracy is the most common metric. It is defiend as the numebr of
%correctly classified samples/ the total number of tested samples
AccuracyNN = sum(comparison)/length(comparison)

% Evaluate accuracy
% Type 1 and Type 2 error
TP= sum ((testLabels==classificationResult) & (testLabels==1))
TN= sum ((testLabels==classificationResult) & (testLabels==-1))
FP= sum ((testLabels~=classificationResult) & (testLabels==-1))
FN= sum ((testLabels~=classificationResult) & (testLabels==1))

% Metrics
Precision = TP / (TP+FP)
Recall = TP / (TP+FN)
Specificity = TN / (TN+FP)
Sensitivity = TP / (TP+FN)
Fmeasure= 2 * TP / (2 * TP+FN+FP)
False_alarm_rate = 1 - Specificity

%% KNN Testing 
[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(hogtrainMatrix);
modelKNN = KNNtraining(Xpca, trainLabels);

for i=1:size(testImages,1)
    testnumber= testHogMatrix(i,:);
    testpca = (testnumber - meanX) * eigenVectors;
    classificationResult(i,1) = KNNTesting(testpca, modelKNN,11);
end

%% Evaluation

% Finally we compared the predicted classification from our mahcine
% learning algorithm against the real labelling of the testing image
comparison = (testLabels==classificationResult);

%Accuracy is the most common metric. It is defiend as the numebr of
%correctly classified samples/ the total number of tested samples
AccuracyKNN = sum(comparison)/length(comparison)

% Evaluate accuracy
% Type 1 and Type 2 error
TP= sum ((testLabels==classificationResult) & (testLabels==1))
TN= sum ((testLabels==classificationResult) & (testLabels==-1))
FP= sum ((testLabels~=classificationResult) & (testLabels==-1))
FN= sum ((testLabels~=classificationResult) & (testLabels==1))

% Metrics
Precision = TP / (TP+FP)
Recall = TP / (TP+FN)
Specificity = TN / (TN+FP)
Sensitivity = TP / (TP+FN)
Fmeasure= 2 * TP / (2 * TP+FN+FP)
False_alarm_rate = 1 - Specificity

%% Notes



% WE NEED TO ADD DETECTOR 








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