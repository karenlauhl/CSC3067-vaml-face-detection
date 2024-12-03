% List of full images for detection
full_image_files = {'im1.jpg', 'im2.jpg', 'im3.jpg', 'im4.jpg'};

% Estimated number of faces in each image 
estimated_face_counts = [7, 15, 8, 57];  

% Sliding window parameters
window_size = [18, 27];   % Set window size to match the training image size (18x27)
step_sizes = [4, 8, 12];  % Different step sizes to test
scales = [1, 0.9, 0.8, 0.7];  % Different scales to detect faces of various sizes

% Metrics counters for all images and step sizes
total_detected_faces = 0;
total_estimated_faces = 0;

% Loop through each full image
for img_index = 1:length(full_image_files)
    % Load the full image (assuming it is already grayscale)
    full_image = imread(full_image_files{img_index});

    % Loop through each step size
    for step_index = 1:length(step_sizes)
        step_size = step_sizes(step_index);
        fprintf('Testing %s with step size: %d\n', full_image_files{img_index}, step_size);

        % Initialize counter for detected boxes
        detected_boxes = [];

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

                    % If the patch is classified as a face, save the detection
                    if prediction == 1
                        original_row = round(row / scales(s));
                        original_col = round(col / scales(s));
                        original_height = round(window_size(1) / scales(s));
                        original_width = round(window_size(2) / scales(s));
                        detected_boxes = [detected_boxes; original_col, original_row, original_width, original_height];
                    end
                end
            end
        end

        % Count the number of detected faces
        num_detected_faces = size(detected_boxes, 1);
        fprintf('Number of detected faces in %s with step size %d: %d\n', ...
                full_image_files{img_index}, step_size, num_detected_faces);

        % Accumulate detected faces and estimated counts
        total_detected_faces = total_detected_faces + num_detected_faces;
        total_estimated_faces = total_estimated_faces + estimated_face_counts(img_index);
    end
end

% Calculate face count accuracy
face_count_accuracy = min(total_detected_faces, total_estimated_faces) / max(total_detected_faces, total_estimated_faces);
fprintf('Face Count Accuracy: %.2f\n', face_count_accuracy);
