function detection = simpleNMS(detection, threshold)
% simpleNMS: Perform non-maxima suppression to remove overlapping bounding boxes
% 
% Inputs:
%   Objects: A matrix where each row represents a detected bounding box [x, y, width, height, confidence]
%   threshold: The intersection-over-union (IoU) threshold for suppression
%
% Output:
%   Objects: The remaining bounding boxes after suppression

% Number of detected objects
numObjects = size(detection, 1);

% Create a vector to store which objects to keep
keep = true(numObjects, 1);

for i = 1:numObjects
    if keep(i) % Only process if this box has not already been removed
        for j = i+1:numObjects
            if keep(j) % Only process if the other box hasn't been removed
                % Calculate intersection area between bounding box i and j
                intersection_area = rectint(detection(i, 1:4), detection(j, 1:4));
                
                % Calculate the area of each bounding box
                box_i_area = detection(i, 3) * detection(i, 4);
                box_j_area = detection(j, 3) * detection(j, 4);
                
                % Calculate the Intersection over Union (IoU)
                iou = intersection_area / (box_i_area + box_j_area - intersection_area);
                
                % If IoU exceeds the threshold, suppress the box with lower confidence
                if iou > threshold
                    if detection(i, 5) > detection(j, 5)
                        keep(j) = false; % Suppress box j
                    else
                        keep(i) = false; % Suppress box i
                    end
                end
            end
        end
    end
end

% Keep only the bounding boxes that were not suppressed
detection = detection(keep, :);

end
