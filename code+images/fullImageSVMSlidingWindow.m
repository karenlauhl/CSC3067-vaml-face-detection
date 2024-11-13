% Step 1: Load a List of Full Images
% List of full images for detection
full_image_files = {'im1.jpg', 'im2.jpg', 'im3.jpg', 'im4.jpg'};

% Sliding window parameters
window_size = [27, 18];   % Set window size to match the training image size (27x18)
step_sizes = [4, 8, 12];  % Different step sizes to test
scales = [1, 0.9, 0.8, 0.7];  % Different scales to detect faces of various sizes

% Loop through each full image
for img_index = 1:length(full_image_files)
    % Load the full image (assuming it is already grayscale)
    full_image = imread(full_image_files{img_index});
    
    % Loop through each step size
    for step_index = 1:length(step_sizes)
        step_size = step_sizes(step_index);
        fprintf('Testing %s with step size: %d\n', full_image_files{img_index}, step_size);

        % Display the image for the current step size
        figure; imshow(full_image); hold on;
        title(['Detections for ', full_image_files{img_index}, ' with Step Size ', num2str(step_size)]);

        % Multi-scale sliding window detection
        for s = 1:length(scales)
            % Resize the full image according to the current scale
            resized_image = imresize(full_image, scales(s));
            [rows, cols] = size(resized_image);

            % Sliding window over the resized image
            for row = 1:step_size:(rows - window_size(1) + 1)
                for col = 1:step_size:(cols - window_size(2) + 1)
                    % Extract the patch of the window size (27x18)
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

        hold off; % Finish drawing detections for this image with the current step size
        pause(1); % Pause to observe the detections before moving to the next step size
    end
end
