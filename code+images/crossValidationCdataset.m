function crossValidationCdataset(cdatasetFile, numFolds, modelType)
    % crossValidateFaceDetection: Perform cross-validation for face detection.
    % Parameters:
    %   cdatasetFile - Path to the .cdataset file (e.g., 'all.cdataset')
    %   numFolds - Number of folds for cross-validation (e.g., 5)
    %   modelType - Type of model to train (e.g., 'tree', 'svm', 'knn')

    % Load images and labels using loadFaceImages
    [images, labels] = loadFaceImages(cdatasetFile);

    % Check the size of the dataset
    fprintf('Loaded %d images with %d labels.\n', size(images, 1), length(labels));

    % Set up stratified cross-validation
    cv = cvpartition(labels, 'KFold', numFolds);

    % Initialize variables for errors
    errors = zeros(cv.NumTestSets, 1);

    % Loop through each fold
    for i = 1:cv.NumTestSets
        % Get training and test indices for this fold
        trainIdx = training(cv, i);
        testIdx = test(cv, i);

        % Split the data into training and testing sets
        XTrain = images(trainIdx, :);
        YTrain = labels(trainIdx);
        XTest = images(testIdx, :);
        YTest = labels(testIdx);

        % Train the model
        model = trainModel(XTrain, YTrain, modelType);

        % Test the model
        YPred = predict(model, XTest);

        % Record error for this fold
        errors(i) = mean(YPred ~= YTest);
    end

    % Calculate overall error
    avgError = mean(errors);

    fprintf('%d-Fold Cross-Validation Error: %.2f%%\n', numFolds, avgError * 100);
end

function model = trainModel(XTrain, YTrain, modelType)
    % Helper function to train a model based on the specified type
    switch lower(modelType)
        case 'tree'
            model = fitctree(XTrain, YTrain); % Decision Tree
        case 'svm'
            model = fitcsvm(XTrain, YTrain); % Support Vector Machine
        case 'knn'
            model = fitcknn(XTrain, YTrain); % k-Nearest Neighbors
        otherwise
            error('Unsupported model type: %s. Use "tree", "svm", or "knn".', modelType);
    end
end
