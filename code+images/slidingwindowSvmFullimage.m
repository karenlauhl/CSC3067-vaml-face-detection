% Step 1: Load a Full Image
% List of full images for detection
full_image_files = {'im1.jpg', 'im2.jpg', 'im3.jpg', 'im4.jpg'};

% Sliding window parameters
window_size = [18, 27];   % Set window size to match the training image size (18x27)
step_size = 8;            % Set step size for moving the window (e.g., 8 pixels)
scales = [1, 0.9, 0.8, 0.7];  % Different scales to detect faces of various sizes

% Loop through each full image
for img_index = 1:length(full_image_files)
    % Load the full image
    full_image = imread(full_image_files{img_index});
    if size(full_image, 3) > 1
        full_image = rgb2gray(full_image);  % Convert to grayscale if necessary
    end

    % Display the image
    figure; imshow(full_image); hold on;

    % Multi-scale sliding window detection
    for s = 1:length(scales)
        % Resize the full image according to the current scale
        resized_image = imresize(full_image, scales(s));
        [rows, cols] = size(resized_image);

        % Sliding window over the resized image
        for row = 1:step_size:(rows - window_size(1) + 1)
            for col = 1:step_size:(cols - window_size(2) + 1)
                % Extract the patch of the window size (18x27)
                window_patch = resized_image(row:(row + window_size(1) - 1), ...
                                             col:(col + window_size(2) - 1));

                % Reshape the patch into a vector for SVM classification
                window_vector = double(reshape(window_patch, 1, []));
                
                % Use the trained SVM model to classify the patch
                [prediction, ~] = SVMTesting(window_vector, model);
                
                % If the patch is classified as a face, mark the detection
                if prediction == 1
                    % Calculate original coordinates for drawing the rectangle on the full image
                    original_row = round(row / scales(s));
                    original_col = round(col / scales(s));
                    original_height = round(window_size(1) / scales(s));
                    original_width = round(window_size(2) / scales(s));
                    
                    % Draw the bounding box on the original image
                    rectangle('Position', [original_col, original_row, original_width, original_height], ...
                              'EdgeColor', 'r', 'LineWidth', 2);
                end
            end
        end
    end

    hold off; % Finish drawing detections for this image
    pause(1); % Pause to observe the detections before moving to the next image
end

if size(full_image, 3) > 1
    full_image = rgb2gray(full_image);  % Convert to grayscale if it's a color image
end

% Step 2: Set Sliding Window Parameters
window_size = [18, 27];   % Set window size to match the size of training images (e.g., 32x32)
step_size = 8;            % Set step size for moving the window (e.g., 8 pixels)
scales = [1, 0.9, 0.8];   % Multi-scale detection to detect faces of different sizes

% Step 3: Prepare to Visualize Detections
figure; imshow(full_image); hold on;

% Step 4: Multi-Scale Sliding Window Detection
for s = 1:length(scales)
    % Resize the full image according to the current scale
    resized_image = imresize(full_image, scales(s));
    [rows, cols] = size(resized_image);

    % Slide window over the resized image
    for row = 1:step_size:(rows - window_size(1) + 1)
        for col = 1:step_size:(cols - window_size(2) + 1)
            % Extract the patch of the window size
            window_patch = resized_image(row:(row + window_size(1) - 1), ...
                                         col:(col + window_size(2) - 1));

            % Reshape the patch into a vector for SVM classification
            window_vector = double(reshape(window_patch, 1, []));
            
            % Classify the patch using the trained SVM model
            [prediction, ~] = SVMTesting(window_vector, model);
            
            % If the patch is classified as a face, mark the detection
            if prediction == 1
                % Calculate original coordinates for drawing the rectangle on the full image
                original_row = round(row / scales(s));
                original_col = round(col / scales(s));
                original_height = round(window_size(1) / scales(s));
                original_width = round(window_size(2) / scales(s));
                
                % Draw the bounding box on the original image
                rectangle('Position', [original_col, original_row, original_width, original_height], ...
                          'EdgeColor', 'r', 'LineWidth', 2);
            end
        end
    end
end

hold off;  % Finish drawing detections
