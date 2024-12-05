% Main workflow script for face detection
clear all
close all
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

ogIm = figure
for i=1:50

    % As you can notice by the size of the matrix image, each digit image
    % has been transform into a long feature vector to be fed in a machine
    % learning algorithm.
    
    %To visualise or recompose the image again, we need to revert that
    %process in its 28x28 image format
    Im = reshape(trainImages(i,:),[27, 18]);
    if trainLabels(i) == 1
        isImage = 'Face';
    else
        isImage = 'Non-Face';
    end
    subplot(5,10,i),colormap gray, imagesc(Im), title(isImage); axis off;
    
end

% Initialize matrix to store HOG features for all training images
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

disp(size(testHogMatrix));

disp('Models trained')

hog1 = figure
imnumber=8;
subplot(1,2,1);
imagesc(reshape(trainImages(imnumber,:),27,18)); axis off;
title('Original Image');    

subplot(1,2,2);
showHog(hogtrainMatrix(imnumber,:),[27, 18]); axis off;
title('HOG Features');


hog2 = figure
imnumber=30;
subplot(1,2,1);
imagesc(reshape(trainImages(imnumber,:),27,18)); axis off;
title('Original Image');    

subplot(1,2,2);
showHog(hogtrainMatrix(imnumber,:),[27, 18]); axis off;
title('HOG Features');


hog3 = figure
imnumber=46;
subplot(1,2,1);
imagesc(reshape(trainImages(imnumber,:),27,18)); axis off;
title('Original Image');    

subplot(1,2,2);
showHog(hogtrainMatrix(imnumber,:),[27, 18]); axis off;
title('HOG Features');

%allImages=[trainImages; testImages];  % , Concatenation horin and ; does it vertical.
%allLabels=[trainLabels; testLabels];  % ; does it vertical.
 
%% Apply PCA
[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(hogtrainMatrix);

%animation for observing the variation of the first eigenvector

eigVector_index=2;
animation = figure;
weights = [-3*sqrt(eigenvalues(eigVector_index)): 6*sqrt(eigenvalues(eigVector_index))/200: 3*sqrt(eigenvalues(eigVector_index))];
for b=weights
    faceReconstruct = meanX + b*eigenVectors(:,eigVector_index)';
    if ishandle(animation)  % Check if the figure handle is still valid
        figure(animation);  
        showHog(faceReconstruct,[27, 18]);
        colormap(gray), axis equal, axis off,
        drawnow
    end
end

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

%% KNN Testing 
modelKNN = KNNtraining(Xpca, trainLabels);

for i=1:size(testImages,1)
    testnumber= testHogMatrix(i,:);
    testpca = (testnumber - meanX) * eigenVectors;
    classificationResult(i,1) = KNNTesting(testpca, modelKNN, 2);
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

%% Detector

%Full image models
full_modelNN = NNtraining(trainImages, trainLabels);
full_modelKNN = KNNtraining(trainImages, trainLabels);
full_modelSVM = SVMtraining(trainImages, trainLabels);

%Hog Feature models
hog_modelNN = NNtraining(hogtrainMatrix, trainLabels);
hog_modelKNN = KNNtraining(hogtrainMatrix, trainLabels);
hog_modelSVM = SVMtraining(hogtrainMatrix, trainLabels);

% Load the image file
groupImage = {'im1.jpg','im2.jpg','im3.jpg','im4.jpg'};

%Run sliding window face detection
finalDetections1 = slidingWindow_full_nn(groupImage, full_modelNN);
finalDetections2 = slidingWindow_full_knn(groupImage, full_modelKNN);
finalDetections3 = slidingWindow_full_Svm(groupImage, full_modelSVM);

finalDetections4 = slidingWindow_hog_nn(groupImage, hog_modelNN);
finalDetections5 = slidingWindow_hog_knn(groupImage, hog_modelKNN);
finalDetections6 = slidingWindow_hog_Svm(groupImage, hog_modelSVM);
