clear;          % Clears all variables from the workspace
clc;            % Clears the command window
close all;      % Closes all open figure windows

% Main workflow script for face detection using full image, NN, sliding window, and NMS

%% Training
% Load training data
[trainImages, trainLabels] = loadFaceImages('face_train.cdataset');   

% Train NN model
modelNN = NNtraining(trainImages, trainLabels);

%% Sliding Window Parameters
windowSize = [27, 18];  % Matching training image dimensions
stepSize = 7;  % Step size for sliding window
scales = [1, 0.9, 0.8, 0.7]; % Multi-scale detection
iouThreshold = 0.3; % IoU threshold for NMS

% List of full images for detection
fullImageFiles = {'im1.jpg'};

%% Detection Loop
for imgIndex = 1:length(fullImageFiles)
    % Load the full image
    fullImage = imread(fullImageFiles{imgIndex});
    figure; imshow(fullImage); hold on;
    title(['Detections for ', fullImageFiles{imgIndex}]);

    % Initialize detections for the current image
    detections = []; % Format: [x, y, width, height, confidence]

    % Loop over scales
    for scale = scales
        resizedImage = imresize(fullImage, scale);
        [rows, cols] = size(resizedImage);

        % Sliding window
        for row = 1:stepSize:(rows - windowSize(1) + 1)
            for col = 1:stepSize:(cols - windowSize(2) + 1)
                patch = resizedImage(row:(row + windowSize(1) - 1), col:(col + windowSize(2) - 1));
                patchVector = double(reshape(patch, 1, []));

                % Classify patch using NN
                prediction = NNTesting(patchVector, modelNN);

                if prediction == 1
                    % Convert coordinates to original scale
                    originalRow = round(row / scale);
                    originalCol = round(col / scale);
                    originalHeight = round(windowSize(1) / scale);
                    originalWidth = round(windowSize(2) / scale);

                    % Append detection
                    detections = [detections; originalCol, originalRow, originalWidth, originalHeight, 1]; % Confidence = 1
                end
            end
        end
    end

    % Apply Non-Maxima Suppression (NMS)
    if exist('applyNMS', 'file')
        detections = applyNMS(detections, iouThreshold);
    end

    % Draw final bounding boxes
    for i = 1:size(detections, 1)
        rectangle('Position', detections(i, 1:4), 'EdgeColor', 'r', 'LineWidth', 2);
    end
    hold off;

    % Optional: Save detections for later evaluation or analysis
    % detectionsOutput{imgIndex} = detections;

end

%% Evaluation with face_test.cdataset
% % Load test patches and their ground truth labels
% [testImages, testLabels] = loadFaceImages('face_test.cdataset');
% 
% % Predict labels for the test patches using the NN model
% classificationResult = [];
% for i = 1:size(testImages, 1)
%     testPatch = reshape(testImages(i, :), [27, 18]); % Reshape test patch
%     prediction = NNTesting(testPatch, modelNN); % Predict using NN model
%     % classificationResult = [classificationResult; prediction]; % Append prediction
%     classificationResult = NNTesting(prediction, modelNN)
% end
% 
% % Check alignment
% if length(testLabels) ~= length(classificationResult)
%     error('testLabels and classificationResult must have the same length.');
% end
% 
% % Finally we compared the predicted classification from our mahcine
% % learning algorithm against the real labelling of the testing image
% comparison = (testLabels==classificationResult);
% 
% %Accuracy is the most common metric. It is defiend as the numebr of
% %correctly classified samples/ the total number of tested samples
% Accuracy = sum(comparison)/length(comparison)
% 
% % Evaluate accuracy
% % Type 1 and Type 2 error
% TP= sum ((testLabels==classificationResult) & (testLabels==1))
% TN= sum ((testLabels==classificationResult) & (testLabels==-1))
% FP= sum ((testLabels~=classificationResult) & (testLabels==-1))
% FN= sum ((testLabels~=classificationResult) & (testLabels==1))
% 
% % Metrics
% Precision = TP / (TP+FP)
% Recall = TP / (TP+FN)
% Specificity = TN / (TN+FP)
% Sensitivity = TP / (TP+FN)
% Fmeasure= 2 * TP / (2 * TP+FN+FP)
% False_alarm_rate = 1 - Specificity
% disp('Face detection complete.');
