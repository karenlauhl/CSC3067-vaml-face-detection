% Main workflow script for face detection

% Load training and test data
% there is total 670= 45*4 + 49*10 in train as calculated
[trainImages, trainLabels] = loadFaceImages('face_train.cdataset');   
[testImages, testLabels] = loadFaceImages('face_test.cdataset');

%allImages=[trainImages; testImages];  % , Concatenation horin and ; does it vertical.
%allLabels=[trainLabels; testLabels];  % ; does it vertical.


%
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
%________________________________________________________________________

%Nn-------------------------
modelNN = NNtraining(hogtrainMatrix, trainLabels);

for i=1:size(testImages,1)
    testnumber= testHogMatrix(i,:);
    classificationResult(i,1) = NNTesting(testnumber, modelNN);
end 

% Evaluate accuracy
accuracy_nn = sum(testLabels==classificationResult) / numel(testLabels);
fprintf('Test Accuracy NN: %.2f%%\n', accuracy_nn * 100);
%_________________________________________________________________________

%KNn-------------------------
modelKNN = KNNtraining(hogtrainMatrix, trainLabels);

for i=1:size(testImages,1)
    testnumber= testHogMatrix(i,:);
    classificationResult(i,1) = KNNTesting(testnumber, modelKNN,11);
end 

% Evaluate accuracy
accuracyKnn = sum(testLabels==classificationResult) / numel(testLabels);
fprintf('Test Accuracy KNN: %.2f%%\n', accuracyKnn * 100);
%_________________________________________________________________________











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