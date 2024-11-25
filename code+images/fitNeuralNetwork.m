function model = fitNeuralNetwork(XTrain, YTrain)
    % Define the neural network structure
    layers = [
        featureInputLayer(size(XTrain, 2))
        fullyConnectedLayer(2) % Binary classification: Face or Non-face
        softmaxLayer
        classificationLayer
    ];

    % Define training options
    options = trainingOptions('sgdm', ...
        'MaxEpochs', 30, ...
        'MiniBatchSize', 16, ...
        'Verbose', false);

    % Train the network
    model = trainNetwork(XTrain, categorical(YTrain), layers, options);
end
