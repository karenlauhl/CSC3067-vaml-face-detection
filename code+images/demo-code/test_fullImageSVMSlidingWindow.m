% List of full images for detection
full_image_files = {'im1.jpg', 'im2.jpg', 'im3.jpg', 'im4.jpg'};

% Updated estimated number of faces in each image (manually counted or estimated)
estimated_face_counts = [7, 15, 8, 57];  % Estimated number of faces in each image

% Sliding window parameters
window_size = [18, 27];   % Set window size to match the training image size (18x27)
step_sizes = [4, 8, 12];  % Different step sizes to test
scales = [1, 0.9, 0.8, 0.7];  % Different scales to detect faces of various sizes

% Initialize an array to store accuracy for all images and step sizes
accuracy_results = zeros(length(full_image_files), length(step_sizes));

% Loop through each full image
for img_index = 1:length(full_image_files)
    % Load the full image (assuming it is already grayscale)
    full_image = imread(full_image_files{img_index});
    estimated_faces = estimated_face_counts(img_index);

    fprintf('\nEvaluating %s (Estimated Faces: %d)\n', full_image_files{img_index}, estimated_faces);

    % Initialize metrics for each step size for this image
    detected_faces_per_step = zeros(length(step_sizes), 1);
    accuracy_per_step = zeros(length(step_sizes), 1);

    % Loop through each step size
    for step_index = 1:length(step_sizes)
        step_size = step_sizes(step_index);
        fprintf('  Testing with step size: %d\n', step_size);

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

        % Count the number of detected faces for this image with the current step size
        num_detected_faces = size(detected_boxes, 1);
        fprintf('    Number of detected faces with step size %d: %d\n', step_size, num_detected_faces);

        % Store the number of detected faces and calculate accuracy for this step size
        detected_faces_per_step(step_index) = num_detected_faces;
        accuracy_per_step(step_index) = min(num_detected_faces, estimated_faces) / max(num_detected_faces, estimated_faces);
        fprintf('    Face Count Accuracy for step size %d: %.2f\n', step_size, accuracy_per_step(step_index));
    end

    % Store accuracy results for this image
    accuracy_results(img_index, :) = accuracy_per_step';

    % Determine the best step size for this image
    [best_accuracy, best_step_index] = max(accuracy_per_step);
    best_step_size = step_sizes(best_step_index);

    fprintf('  Best step size for %s is %d with an accuracy of %.2f\n', ...
            full_image_files{img_index}, best_step_size, best_accuracy);
end

% Plotting the results
figure;
for img_index = 1:length(full_image_files)
    subplot(2, 2, img_index);
    bar(step_sizes, accuracy_results(img_index, :));
    title(['Accuracy for ', full_image_files{img_index}]);
    xlabel('Step Size');
    ylabel('Face Count Accuracy');
    ylim([0 1]);
    grid on;
end
