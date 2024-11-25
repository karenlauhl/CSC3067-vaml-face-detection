function modelNN = NNtraining(images, labels)
    % Normalize the images (e.g., scale to [0, 1])
    images = images ./ max(images(:));
    
    % Store training data in the model
    modelNN.neighbours = images;
    modelNN.labels = labels;
end
