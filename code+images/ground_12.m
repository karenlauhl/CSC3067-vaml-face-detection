clear currentGroundTruthBoxes image imageFiles iouThreshold ;
% File names for images
imageFiles = {'im1.jpg', 'im2.jpg', 'im3.jpg', 'im4.jpg'};

% Manually defined ground truth bounding boxes for each image
% Format: [x, y, width, height]
groundTruthBoxes.im4 = [
    9, 21, 17, 26;
    25, 20, 18, 21;
    47, 29, 19, 21;
    65, 17, 18, 23;
    83, 28, 17, 22;
    99, 14, 17, 22;
    120, 24, 18, 27
    ];

groundTruthBoxes.im3 = [
    7, 9, 18, 23;
    32, 9, 18, 22;
    57, 10, 18, 21;
    82, 9, 18, 22;
    106, 9, 19, 22;
    6, 37, 19, 25;
    30, 37, 19, 25;
    56, 37, 19, 24;
    80, 37, 20, 24;
    105, 37, 20, 24;
    6, 68, 18, 23;
    32, 68, 17, 23;
    56, 69, 19, 23;
    81, 70, 17, 21;
    106, 72, 18, 19
    ];
groundTruthBoxes.im2 = [
    13, 32, 22, 22;
    38, 46, 18, 22;
    28, 77, 17, 22;
    66, 54, 14, 19;
    93, 78, 15, 15;
    117, 59, 19, 20;
    95, 34, 17, 21;
    70, 21, 15, 19
    ];

groundTruthBoxes.im1 = [
    20, 189, 19, 29;
    26, 117, 20, 24;
    112, 188, 24, 25;
    206, 179, 24, 27;
    302, 186, 21, 28;
    374, 187, 21, 29;
    453, 181, 22, 29;
    532, 194, 21, 26;
    551, 144, 20, 29;
    511, 118, 23, 26;
    471, 147, 22, 31;
    450, 128, 20, 21;
    392, 144, 17, 25;
    363, 130, 21, 25;
    337, 149, 22, 25;
    519, 74, 21, 27;
    496, 98, 16, 22;
    454, 66, 19, 25;
    441, 93, 18, 21;
    489, 42, 17, 24;
    467, 30, 16, 24;
    438, 46, 17, 22;
    408, 29, 15, 22;
    379, 30, 19, 21;
    409, 63, 17, 22;
    383, 65, 15, 21;
    320, 24, 19, 19;
    335, 59, 17, 22;
    364, 78, 18, 22;
    387, 112, 18, 25;
    323, 89, 20, 25;
    292, 73, 16, 23;
    267, 25, 16, 22;
    284, 50, 15, 21;
    304, 129, 22, 25;
    272, 141, 20, 26;
    256, 98, 21, 27;
    210, 109, 21, 28;
    256, 64, 16, 28;
    236, 46, 14, 21;
    205, 21, 18, 25;
    156, 48, 21, 30;
    204, 61, 22, 22;
    191, 83, 22, 26;
    177, 114, 18, 24;
    145, 119, 20, 24;
    141, 154, 19, 25;
    141, 84, 19, 23;
    140, 27, 20, 27;
    101, 42, 23, 27;
    109, 88, 18, 23;
    88, 120, 23, 20;
    80, 143, 19, 26;
    49, 85, 23, 26;
    67, 32, 17, 20;
    42, 54, 18, 21;
    65, 71, 17, 22
    ];

% Explicit mapping between images and their corresponding ground truth boxes
imageToGroundTruthMap = {
    'im1.jpg', 'im1';
    'im2.jpg', 'im2';
    'im3.jpg', 'im3';
    'im4.jpg', 'im4'
    };


% Parameters
iouThreshold = 0.3; % IoU threshold for determining a match
outputDir = 'output_images';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end


% Initialize metrics
TP = 0; FP = 0; FN = 0; TN = 0;

% Loop through each image
for i = 1:length(imageFiles)
    % Load the current image
    image = imread(imageFiles{i});

    % Map the current image to the correct ground truth key
    % Find the ground truth mapping for the current image
    rowIndex = strcmp(imageToGroundTruthMap(:, 1), imageFiles{i});
    if ~any(rowIndex)
        error(['No ground truth mapping found for ', imageFiles{i}]);
    end
    imageName = imageToGroundTruthMap{rowIndex, 2};


    % Fetch the corresponding ground truth boxes
    if ~isfield(groundTruthBoxes, imageName)
        error(['Ground truth for ', imageName, ' is not defined.']);
    end
    currentGroundTruthBoxes = groundTruthBoxes.(imageName);

    % Debugging: Check the mapping
    disp(['Processing image: ', imageFiles{i}]);
    disp(['Using ground truth for: ', imageName]);
    disp('Ground truth boxes:');
    disp(currentGroundTruthBoxes);

    % Check for valid predictions
    if isempty(finalDetections12)
        disp(['No faces detected in ', imageFiles{i}]);
        continue;
    end


    % Compare predicted boxes with ground truth
    matchedGroundTruth = false(size(currentGroundTruthBoxes, 1), 1); % Track matched ground truth boxes
    for j = 1:size(finalDetections12, 1)
        predBox = finalDetections12(j, 1:4); % Ensure correct format for prediction box
        maxIoU = 0; % Initialize maximum IoU
        bestMatchIdx = 0; % Track best match index

        % Compare the predicted box with each ground truth box
        for k = 1:size(currentGroundTruthBoxes, 1)
            gtBox = currentGroundTruthBoxes(k, :); % Get ground truth box

            % Compute IoU
            iou = calculateIoU(predBox, gtBox);

            % Update max IoU and match index if conditions are met
            if iou > maxIoU && iou >= iouThreshold && ~matchedGroundTruth(k)
                maxIoU = iou;
                bestMatchIdx = k;
            end
        end

        % Determine if the predicted box is a True Positive or False Positive
        if maxIoU >= iouThreshold
            % True Positive: Match found with ground truth
            TP = TP + 1;
            matchedGroundTruth(bestMatchIdx) = true; % Mark the ground truth as matched
            rectangle('Position', predBox, 'EdgeColor', 'g', 'LineWidth', 2); % Green: TP
        else
            % False Positive: No match found
            FP = FP + 1;
            rectangle('Position', predBox, 'EdgeColor', 'r', 'LineWidth', 2); % Red: FP
        end
    end

    % Count False Negatives
    for k = 1:size(currentGroundTruthBoxes, 1)
        if ~matchedGroundTruth(k)
            FN = FN + 1;
            rectangle('Position', currentGroundTruthBoxes(k, :), 'EdgeColor', 'b', 'LineWidth', 2); % Blue: FN
        end
    end


    % Save output image
    saveas(gcf, fullfile(outputDir, ['output_' imageFiles{i}]));

    % Close figure
    close;
end

% Display metrics
disp('Performance Metrics:');
disp(['True Positives (TP): ', num2str(TP)]);
disp(['False Positives (FP): ', num2str(FP)]);
disp(['False Negatives (FN): ', num2str(FN)]);

% Compute Precision, Recall, and F-measure
precision = TP / (TP + FP);
recall = TP / (TP + FN);
fmeasure = 2 * (precision * recall) / (precision + recall);
disp(['Precision: ', num2str(precision)]);
disp(['Recall: ', num2str(recall)]);
disp(['F-measure: ', num2str(fmeasure)]);


disp('Detection and evaluation complete.');

% Helper Function: Calculate IoU
function iou = calculateIoU(boxA, boxB)
% Validate dimensions
if numel(boxA) ~= 4 || numel(boxB) ~= 4
    error('Bounding boxes must have 4 elements: [x, y, width, height].');
end

% Calculate Intersection Over Union
xA = max(boxA(1), boxB(1));
yA = max(boxA(2), boxB(2));
xB = min(boxA(1) + boxA(3), boxB(1) + boxB(3));
yB = min(boxA(2) + boxA(4), boxB(2) + boxB(4));

intersection = max(0, xB - xA) * max(0, yB - yA);
areaA = boxA(3) * boxA(4);
areaB = boxB(3) * boxB(4);

union = areaA + areaB - intersection;
if union <= 0
    iou = 0; % Avoid division by zero
else
    iou = intersection / union;
end
end