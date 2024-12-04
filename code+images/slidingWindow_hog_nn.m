function [detections] = slidingWindow_hog_nn(fullImageFiles, model)
    % Define parameters
    windowSize = [27, 18];
    stepSize = 7;
    scales = [1,0.9, 0.8, 0.7]; % Fixed scale value as per your request


 for imgIndex = 1:length(fullImageFiles)
    % Load the full image
    fullImage = imread(fullImageFiles{imgIndex});
    figure;
    imshow(fullImage);
    hold on;
    title(['Detections for(HOG & NN) ', fullImageFiles{imgIndex}]);

    % Initialize variables for storing detections
    detections = [];

    % Loop over each scale in the image pyramid
    for scale = scales
        % Resize the image based on current scale
        resizedImg = imresize(fullImage, scale);
        [rows, cols] = size(resizedImg);

        % Sliding window over the scaled image
        for row = 1:stepSize:(rows - windowSize(1) + 1)
            for col = 1:stepSize:(cols - windowSize(2) + 1)
                % Extract the window
                window_mat = resizedImg(row:(row + windowSize(1) - 1), col:(col + windowSize(2) - 1));
                window_mat = reshape(window_mat,1, []); 
                % Normalize first do in after loading dataset
                window_mat = double(window_mat);
                window_mat = window_mat / 255;
                % Reshape the window to a 2D image for feature extraction
                window_mat = reshape(window_mat, 27, 18); 
                 
                % Extract HOG features
                hogFeatures = hog_feature_vector(window_mat);

                % Use KNN to classify the extracted features
                predictedLabel = NNTesting(hogFeatures, model);
                
                % If a face is detected (predictedLabel == 1), store the detection
                if predictedLabel == 1
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

    % Print the total number of detections before NMS
    fprintf('Total number of detections before NMS: %d\n', size(detections, 1));
    
    % Apply Non-Maximum Suppression (NMS) to reduce false positives
    detections = applyNMS(detections, 0.3);

    % Draw final bounding boxes
    for i = 1:size(detections, 1)
        rectangle('Position', detections(i, 1:4), 'EdgeColor', 'r', 'LineWidth', 2);
    end
    hold off;

    % Print the total number of detections after NMS
    fprintf('Total number of detections: %d\n', size(detections, 1));
 end


end