function [detections, outputImage] = slidingWindow_hog_knn(image, model)
    % Define parameters
    windowSize = [27, 18];
    stepSize = 3;
    scales = [1]; % Fixed scale value as per your request

    % Initialize variables for storing detections
    detections = [];
    outputImage = image; % Copy of the image for visualization

    % Loop over each scale in the image pyramid
    for scale = scales
        % Resize the image based on current scale
        resizedImg = imresize(image, scale);
        [height, width] = size(resizedImg);

        % Sliding window over the scaled image
        for x = 1:stepSize:(width - windowSize(2) + 1)
            for y = 1:stepSize:(height - windowSize(1) + 1)
                % Extract the window
                window_mat = resizedImg(y:y+windowSize(1)-1, x:x+windowSize(2)-1);
                window_mat = reshape(window_mat,1, 27*18); 
                % Normalize first do in after loading dataset
                window_mat = double(window_mat);
                window_mat = window_mat / 255;
                % Reshape the window to a 2D image for feature extraction
                window_mat = reshape(window_mat, 27, 18); 
                 
                % Extract HOG features
                hogFeatures = hog_feature_vector(window_mat);

                % Use KNN to classify the extracted features
                predictedLabel = KNNTesting(hogFeatures, model, 11);
                
                % If a face is detected (predictedLabel == 1), store the detection
                if predictedLabel == 1
                    % Store as [x, y, width, height, scale]
                    detections = [detections; x, y, windowSize(2), windowSize(1), scale];
                end
            end
        end
    end

    % Print the total number of detections before NMS
    fprintf('Total number of detections before NMS: %d\n', size(detections, 1));
    
    % Apply Non-Maximum Suppression (NMS) to reduce false positives
    detections = applyNMS(detections, 0.6);

    % Visualize the detections by drawing rectangles on the original image
    figure;
    imshow(outputImage); hold on;
    for i = 1:size(detections, 1)
        x = detections(i, 1);
        y = detections(i, 2);
        width = detections(i, 3);
        height = detections(i, 4);
        scale = detections(i, 5);

        % Scale the window size back to original image coordinates
        rectPosition = [x / scale, y / scale, width / scale, height / scale];

        % Draw rectangle
        rectangle('Position', rectPosition, 'EdgeColor', 'r', 'LineWidth', 1.5);
    end
    hold off;

    % Print the total number of detections after NMS
    fprintf('Total number of detections: %d\n', size(detections, 1));
end
