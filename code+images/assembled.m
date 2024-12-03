% Main workflow script for face detection
%% Training

% Load training and test data
% there is total 670= 45*4 + 49*10 in train as calculated
[trainImages, trainLabels] = loadFaceImages('face_train.cdataset');   
[testImages, testLabels] = loadFaceImages('face_test.cdataset');
% Normalize
trainImages = double(trainImages);
testImages = double(testImages);
trainImages=trainImages/255;
testImages=testImages/255;

%allImages=[trainImages; testImages];  % , Concatenation horin and ; does it vertical.
%allLabels=[trainLabels; testLabels];  % ; does it vertical.

% Initialize matrix to store HOG features for all TRAINING images
numImages = size(trainImages, 1); % Number of images in the dataset
hogtrainMatrix = zeros(numImages, length(hog_feature_vector(reshape(trainImages(1, :), 27, 18)))); % Preallocate
% Loop through each image in the training set
for i = 1:numImages
    % Reshape each image from vector to 2D matrix
    imgt = reshape(trainImages(i, :),27, 18);
    % Extract HOG features for the current image
    hogtrainMatrix(i, :) = hog_feature_vector(imgt); % Store HOG features in the matrix
end
%===============
% Initialize matrix for HOG features of TEST images
numTestImages = size(testImages, 1);
testHogMatrix = zeros(numTestImages, size(hogtrainMatrix, 2)); % Preallocate
% Loop through the test images and extract HOG features
for i = 1:numTestImages
    % Reshape the test image to 2D
    testImg = reshape(testImages(i, :), [27, 18]);
    % Extract HOG features for the test image
    testHogMatrix(i, :) = hog_feature_vector(testImg); % Store HOG features
end
%________________________________________________________________________

% Apply PCA
[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(hogtrainMatrix);

%% NN Model Training
modelNN = NNtraining(Xpca, trainLabels);
%% NN Testing

for i=1:size(testImages,1)
    testnumber= testHogMatrix(i,:);
    testpca = (testnumber - meanX) * eigenVectors;
    classificationResult(i,1) = NNTesting(testpca, modelNN);
end

%% Evaluation
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

% Metrics
Precision = TP / (TP+FP)
Recall = TP / (TP+FN)
Specificity = TN / (TN+FP)
Sensitivity = TP / (TP+FN)
Fmeasure= 2 * TP / (2 * TP+FN+FP)
False_alarm_rate = 1 - Specificity

%% KNN Testing 
modelKNN = KNNtraining(Xpca, trainLabels);

for i=1:size(testImages,1)
    testnumber= testHogMatrix(i,:);
    testpca = (testnumber - meanX) * eigenVectors;
    classificationResult(i,1) = KNNTesting(testpca, modelKNN,11);
end

%% Evaluation
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

% Metrics
Precision = TP / (TP+FP)
Recall = TP / (TP+FN)
Specificity = TN / (TN+FP)
Sensitivity = TP / (TP+FN)
Fmeasure= 2 * TP / (2 * TP+FN+FP)
False_alarm_rate = 1 - Specificity

%% SVM Model Training 
modelSVM = SVMtraining(Xpca, trainLabels);

for i=1:size(testImages,1)
    testnumber= testHogMatrix(i,:);
    testpca = (testnumber - meanX) * eigenVectors;
    classificationResult(i,1) = SVMTesting(testpca, modelSVM);
end

%% Evaluation
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

% Metrics
Precision = TP / (TP+FP)
Recall = TP / (TP+FN)
Specificity = TN / (TN+FP)
Sensitivity = TP / (TP+FN)
Fmeasure= 2 * TP / (2 * TP+FN+FP)
False_alarm_rate = 1 - Specificity

%% Notes

% WE NEED TO ADD DETECTOR 

% Load the image file
groupImage = {'im1.jpg','im2.jpg'};


full_modelNN = NNtraining(trainImages, trainLabels);
full_modelKNN = KNNtraining(trainImages, trainLabels);
%Run sliding window face detection
finalDetections1 = slidingWindow_full_nn(groupImage, full_modelNN);
finalDetections2 = slidingWindow_full_knn(groupImage, full_modelKNN);

%finalDetections3 = slidingWindow_hog_nn(groupImage, hog_modelNN);
%finalDetections4 = slidingWindow_hog_knn(groupImage, hog_modelKNN);

