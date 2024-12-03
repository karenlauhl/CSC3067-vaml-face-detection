clear; clc; close all;  % Clear workspace, command window, and close figures

% Main workflow script for face detection using KNN and evaluation with face_test.cdataset

%% Training
[trainImages, trainLabels] = loadFaceImages('face_train.cdataset');   
modelKNN.neighbours = trainImages;  % Store training images in the model
modelKNN.labels = trainLabels;     % Store training labels in the model
K = 5;  % Number of nearest neighbors

%% Sliding Window Parameters
windowSize = [27, 18];
stepSize = 7;
scales = [1, 0.9, 0.8, 0.7];
iouThreshold = 0.3;

fullImageFiles = {'im1.jpg'};

%% Detection Loop (Optional: For visualizing detections)
for imgIndex = 1:length(fullImageFiles)
    fullImage = imread(fullImageFiles{imgIndex});
    figure; imshow(fullImage); hold on;
    title(['Detections for ', fullImageFiles{imgIndex}]);

    detections = [];
    for scale = scales
        resizedImage = imresize(fullImage, scale);
        [rows, cols] = size(resizedImage);

        for row = 1:stepSize:(rows - windowSize(1) + 1)
            for col = 1:stepSize:(cols - windowSize(2) + 1)
                patch = resizedImage(row:(row + windowSize(1) - 1), col:(col + windowSize(2) - 1));
                patchVector = double(reshape(patch, 1, []));
                prediction = KNNTesting(patchVector, modelKNN, K);  % Use the declared K

                if prediction == 1
                    originalRow = round(row / scale);
                    originalCol = round(col / scale);
                    originalHeight = round(windowSize(1) / scale);
                    originalWidth = round(windowSize(2) / scale);
                    detections = [detections; originalCol, originalRow, originalWidth, originalHeight, 1];
                end
            end
        end
    end

    if exist('applyNMS', 'file')
        detections = applyNMS(detections, iouThreshold);
    end

    for i = 1:size(detections, 1)
        rectangle('Position', detections(i, 1:4), 'EdgeColor', 'r', 'LineWidth', 2);
    end
    hold off;
end

%% Evaluation with face_test.cdataset
[testImages, testLabels] = loadFaceImages('face_test.cdataset');

classificationResult = [];
for i = 1:size(testImages, 1)
    testPatch = testImages(i, :);
    prediction = KNNTesting(testPatch, modelKNN, K);  % Use the declared K
    classificationResult = [classificationResult; prediction];
end

if length(testLabels) ~= length(classificationResult)
    error('testLabels and classificationResult must have the same length.');
end

comparison = (testLabels == classificationResult);
Accuracy = sum(comparison) / length(comparison);

TP = sum((testLabels == classificationResult) & (testLabels == 1));
TN = sum((testLabels == classificationResult) & (testLabels == -1));
FP = sum((testLabels ~= classificationResult) & (testLabels == -1));
FN = sum((testLabels ~= classificationResult) & (testLabels == 1));

Precision = TP / (TP + FP);
Recall = TP / (TP + FN);
Specificity = TN / (TN + FP);
Fmeasure = 2 * (Precision * Recall) / (Precision + Recall);
False_alarm_rate = 1 - Specificity;

disp('Evaluating KNN Model:');
disp(['Accuracy: ', num2str(Accuracy)]);
disp(['True Positives (TP): ', num2str(TP)]);
disp(['True Negatives (TN): ', num2str(TN)]);
disp(['False Positives (FP): ', num2str(FP)]);
disp(['False Negatives (FN): ', num2str(FN)]);
disp(['Precision: ', num2str(Precision)]);
disp(['Recall: ', num2str(Recall)]);
disp(['F-measure: ', num2str(Fmeasure)]);
disp(['Specificity: ', num2str(Specificity)]);
disp(['False Alarm Rate: ', num2str(False_alarm_rate)]);
disp('Face detection complete.');
