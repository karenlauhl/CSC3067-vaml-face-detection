% Function to apply Non-Maximum Suppression
function filteredDetections = applyNMS(detections, iouThreshold)
    if isempty(detections)
        filteredDetections = [];
        return;
    end

    % Initialize an empty array for the final boxes
    filteredDetections = [];
    % Sort detections by their confidence (here assumed to be the scale)
    [~, idx] = sort(detections(:, 5), 'descend');
    detections = detections(idx, :);  % Sort detections by scale (confidence)

    while ~isempty(detections)
        % Take the top detection (highest score)
        topDetection = detections(1, :);
        filteredDetections = [filteredDetections; topDetection];
        
        % Remove the top detection from the list
        detections(1, :) = [];
        
        % Compute IoU for remaining boxes and suppress those with high overlap
        iou = computeIoU(topDetection, detections);
        detections(iou > iouThreshold, :) = [];
    end
end

% Function to compute IoU (Intersection over Union)
function iou = computeIoU(box1, boxes)
    x1 = box1(1);
    y1 = box1(2);
    w1 = box1(3);
    h1 = box1(4);

    x2 = boxes(:, 1);
    y2 = boxes(:, 2);
    w2 = boxes(:, 3);
    h2 = boxes(:, 4);

    % Compute the area of intersection
    x1_inter = max(x1, x2);
    y1_inter = max(y1, y2);
    x2_inter = min(x1 + w1, x2 + w2);
    y2_inter = min(y1 + h1, y2 + h2);

    % Compute the area of intersection (use .* for element-wise multiplication)
    interArea = max(0, x2_inter - x1_inter) .* max(0, y2_inter - y1_inter);
    
    % Compute the area of both boxes
    area1 = w1 .* h1;  % Use .* for element-wise multiplication
    area2 = w2 .* h2;  % Use .* for element-wise multiplication

    % Compute IoU
    unionArea = area1 + area2 - interArea;
    iou = interArea ./ unionArea;
end

