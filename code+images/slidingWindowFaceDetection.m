function [detections, outputImage] = slidingWindowFaceDetection(image, model, windowSize, stepSize, scales, iouThreshold)
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
                window = resizedImg(y:y+windowSize(1)-1, x:x+windowSize(2)-1);

                % Flatten window to raw pixel values
                XFeatures = double(window(:))'; % Convert to a row vector of pixel intensities

                % Ensure XFeatures matches the model's expected feature dimension
                expectedDim = size(model.X, 2); % Number of features used during training
                if length(XFeatures) < expectedDim
                    XFeatures = [XFeatures, zeros(1, expectedDim - length(XFeatures))];
                elseif length(XFeatures) > expectedDim
                    XFeatures = XFeatures(1:expectedDim);
                end

                % Predict using the trained model
                [predictedLabel, ~] = predict(model, XFeatures);

                % If a face is detected, store the coordinates, width, height, and scale
                if (predictedLabel == 1)
                    % Store as [x, y, width, height, scale]
                    detections = [detections; x, y, windowSize(2), windowSize(1), scale];
                end
            end
        end
    end

    % Apply Non-Maximum Suppression (NMS) to reduce false positives
    detections = applyNMS(detections, iouThreshold);

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

    % Print the total number of detections
    fprintf('Total number of detections: %d\n', size(detections, 1));
    
end
