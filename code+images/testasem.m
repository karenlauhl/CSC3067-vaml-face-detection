% Main workflow script for face detection

% Load training and test data
% there is total 670= 45*4 + 49*10 in train as calculated
[trainImages, trainLabels] = loadFaceImages('face_train.cdataset');   
[testImages, testLabels] = loadFaceImages('face_test.cdataset');
% Normalize
trainImages = double(trainImages);
testImages = double(testImages);
trainImages=trainImages/255;
testImages=testImages/255;  %images are in vector form here.

%allImages=[trainImages; testImages];  % , Concatenation horin and ; does it vertical.
%allLabels=[trainLabels; testLabels];  % ; does it vertical.

%HOG extraction
%========================================================================
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
%---------------------------
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
%===========================================================================================

%Gabor Extraction =======================================================
% Number of training samples (assuming you have a list of images and labels)
numImages = size(trainImages, 1);
trainGFeatures = zeros(numImages, 19440);  % Assuming 19440 features per sample
for i = 1:numImages
     imgt = reshape(trainImages(i, :),27, 18);  % Load your training image
    trainGFeatures(i, :) = gabor_feature_vector(imgt);  % Extract Gabor feature vector
end
%===========================================================================================


% %Nn-------------------------
% hog_modelNN = NNtraining(hogtrainMatrix, trainLabels);
% 
% for i=1:size(testImages,1)
%     testnumber= testHogMatrix(i,:);
%     classificationResult(i,1) = NNTesting(testnumber, hog_modelNN);
% end 
% 
% % Evaluate accuracy
% accuracy_nn = sum(testLabels==classificationResult) / numel(testLabels);
% fprintf('Test Accuracy NN: %.2f%%\n', accuracy_nn * 100);
% %_________________________________________________________________________
% 
% %KNn-------------------------
% hog_modelKNN = KNNtraining(hogtrainMatrix, trainLabels);
% %save detectorModelknn modelKNN;
% 
% for i=1:size(testImages,1)
%     testnumber= testHogMatrix(i,:);
%     classificationResult(i,1) = KNNTesting(testnumber, hog_modelKNN,11);
% end 
% 
% % Evaluate accuracy
% accuracyKnn = sum(testLabels==classificationResult) / numel(testLabels);
% fprintf('Test Accuracy KNN: %.2f%%\n', accuracyKnn * 100);
% %_________________________________________________________________________

% %SVM-------------------------
% hog_modelSVM = SVMtraining(hogtrainMatrix, trainLabels);
% 
% for i=1:size(testImages,1)
%     testnumber= testHogMatrix(i,:);
%     classificationResult(i,1) = SVMTesting(testnumber, hog_modelSVM);
% end 
% 
% % Evaluate accuracy
% accuracySVM = sum(testLabels==classificationResult) / numel(testLabels);
% fprintf('Test Accuracy SVM: %.2f%%\n', accuracySVM * 100);
% %_________________________________________________________________________

% Apply PCA on Gabor for traininig and have to take eigenvalues and from here to testinf ===================
% Standardize the Gabor features
GfeatureMean = mean(trainGFeatures);
GfeatureStd = std(trainGFeatures);
GtrainMatrix_scaled = (trainGFeatures - GfeatureMean) ./ GfeatureStd;

% Perform PCA on the scaled features
[GeigenVectors, Geigenvalues, GmeanX, GabortrainMatrixPCA] = PrincipalComponentAnalysis(GtrainMatrix_scaled,100);
%=========================================================================================================

% Apply PCA on HOG for traininig and have to take eigenvalues and from here to testinf ===================
% Standardize the HOG features
HfeatureMean = mean(hogtrainMatrix);
HfeatureStd = std(hogtrainMatrix);
hogtrainMatrix_scaled = (hogtrainMatrix - HfeatureMean) ./ HfeatureStd;

% Perform PCA on the scaled features
[hogeigenVectors, hogeigenvalues, hogmeanX, hogtrainMatrixPCA] = PrincipalComponentAnalysis(hogtrainMatrix_scaled);
%=============================================================================================================




%XPCA with hog Models=============================================================================
Hpca_modelNN = NNtraining(hogtrainMatrixPCA, trainLabels);
Hpca_modelKNN = KNNtraining(hogtrainMatrixPCA, trainLabels);
Hpca_modelSVM = SVMtraining(hogtrainMatrixPCA, trainLabels);

%Full image models
full_modelNN = NNtraining(trainImages, trainLabels);
full_modelKNN = KNNtraining(trainImages, trainLabels);
full_modelSVM = SVMtraining(trainImages, trainLabels);

%Hog Feature models
hog_modelNN = NNtraining(hogtrainMatrix, trainLabels);
hog_modelKNN = KNNtraining(hogtrainMatrix, trainLabels);
hog_modelSVM = SVMtraining(hogtrainMatrix, trainLabels);

%Gabor Feature models
gabor_modelNN = NNtraining(trainGFeatures, trainLabels);
gabor_modelKNN = KNNtraining(trainGFeatures, trainLabels);
gabor_modelSVM = SVMtraining(trainGFeatures, trainLabels);

%XPCA with gabor Models
% Gpca_modelNN = NNtraining(GabortrainMatrixPCA, trainLabels);
% Gpca_modelKNN = KNNtraining(GabortrainMatrixPCA, trainLabels);
% Gpca_modelSVM = SVMtraining(GabortrainMatrixPCA, trainLabels);


% Cell array containing image file names
groupImage = {'im1.jpg', 'im2.jpg','im3.jpg','im4.jpg'};


%Run sliding window face detection
  % finalDetections1 = slidingWindow_full_nn(groupImage, full_modelNN);
  % finalDetections2 = slidingWindow_full_knn(groupImage, full_modelKNN);
  % finalDetections3 = slidingWindow_full_Svm(groupImage, full_modelSVM);
  % 
  % finalDetections4 = slidingWindow_hog_nn(groupImage, hog_modelNN);
  % finalDetections5 = slidingWindow_hog_knn(groupImage, hog_modelKNN);
  % finalDetections6 = slidingWindow_hog_Svm(groupImage, hog_modelSVM);
   % 
   % finalDetections10 = slidingWindow_pca_knn(groupImage, Hpca_modelKNN,hogeigenVectors,hogmeanX,HfeatureMean,HfeatureStd);
   % finalDetections11 = slidingWindow_pca_nn(groupImage, Hpca_modelNN,hogeigenVectors,hogmeanX,HfeatureMean,HfeatureStd);
   % finalDetections12 = slidingWindow_pca_svm(groupImage, H2pca_modelSVM,hogeigenVectors,hogmeanX,HfeatureMean,HfeatureStd);

   % finalDetections7 = slidingWindow_gabor_nn(groupImage, gabor_modelNN);
   % finalDetections8 = slidingWindow_gabor_knn(groupImage, gabor_modelKNN);
   % finalDetections9 = slidingWindow_gabor_Svm(groupImage, gabor_modelSVM);

   % finalDetections13 = slidingWindow_gabor_pca_knn(groupImage, Gpca_modelKNN,GeigenVectors,GmeanX,GfeatureMean,GfeatureStd);
   % finalDetections14 = slidingWindow_gabor_pca_nn(groupImage, Gpca_modelNN,GeigenVectors,GmeanX,GfeatureMean,GfeatureStd);
   % finalDetections15 = slidingWindow_gabor_pca_svm(groupImage, Gpca_modelSVM,GeigenVectors,GmeanX,GfeatureMean,GfeatureStd);

