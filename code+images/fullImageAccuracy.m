% List of full images for detection
full_image_files = {'im1.jpg', 'im2.jpg', 'im3.jpg', 'im4.jpg'};

% Ground truth bounding boxes for each image (manually defined)
% Each row represents [x, y, width, height] for a face in the image
ground_truth_boxes = {
    [50, 30, 18, 27; 120, 80, 18, 27],    % Ground truth boxes for im1.jpg
    [40, 20, 18, 27],                     % Ground truth boxes for im2.jpg
    [70, 50, 18, 27; 200, 100, 18, 27],   % Ground truth boxes for im3.jpg
    [90, 70, 18, 27]                      % Ground truth boxes for im4.jpg
};

% Sliding window parameters
window_size = [18, 27];   % Set window size to match the training image size (18x27)
step_size = 8;            % Set step size for moving the window (e.g., 8 pixels)
scales = [1, 0.9, 0.8, 0.7];  % Different scales to detect faces of various sizes

% Metrics counters
TP = 0; FP = 0; FN = 0;

% Loop through each full image
for img_index = 1:length(full_image_files)
    % Load the full image
    full_image = imread(full_image_files{img_index});
    if size(full_image, 3) > 1
        full_image = rgb2gray(full_image);  % Convert to grayscale if necessary
    end

    % Get ground truth boxes for the current image
    gt_boxes = ground_truth_boxes{img_index};
    num_gt_boxes = size(gt_boxes, 1);

    % Initialize list for detected boxes
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

    % Calculate IoU and classify detections
    matched_gt = zeros(num_gt_boxes, 1); % Track matched ground truth boxes
    for i = 1:size(detected_boxes, 1)
        detected_box = detected_boxes(i, :);
        max_iou = 0;
        max_iou_idx = -1;

        % Calculate IoU with each ground truth box
        for j = 1:num_gt_boxes
            gt_box = gt_boxes(j, :);
            iou = calculateIoU(detected_box, gt_box);
            if iou > max_iou
                max_iou = iou;
                max_iou_idx = j;
            end
        end

        % Consider a detection correct if IoU > 0.5
        if max_iou > 0.5 && matched_gt(max_iou_idx) == 0
            TP = TP + 1;
            matched_gt(max_iou_idx) = 1; % Mark the ground truth as matched
        else
            FP = FP + 1; % False Positive if IoU < 0.5 or duplicate detection
        end
    end

    % Any unmatched ground truth boxes are False Negatives
    FN = FN + sum(matched_gt == 0);
end

% Calculate Precision, Recall, and F1 Score
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1_score = 2 * (precision * recall) / (precision + recall);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1 Score: %.2f\n', f1_score);

% Helper Function to Calculate IoU
function iou = calculateIoU(boxA, boxB)
    % Calculate coordinates of the intersection rectangle
    xA = max(boxA(1), boxB(1));
    yA = max(boxA(2), boxB(2));
    xB = min(boxA(1) + boxA(3), boxB(1) + boxB(3));
    yB = min(boxA(2) + boxA(4), boxB(2) + boxB(4));

    % Compute the area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1);

    % Compute the area of both the prediction and ground-truth rectangles
    boxAArea = boxA(3) * boxA(4);
    boxBArea = boxB(3) * boxB(4);

    % Compute the intersection over union
    iou = interArea / (boxAArea + boxBArea - interArea);
end
