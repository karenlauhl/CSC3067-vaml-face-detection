function predictions = predictNeuralNetwork(model, XTest)
    % Predict using the neural network
    probabilities = predict(model, XTest);
    [~, predictions] = max(probabilities, [], 2); % Class with highest probability
    predictions = predictions - 1; % Adjust to match label format
end
