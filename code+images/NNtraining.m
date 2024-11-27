function modelNN = NNtraining(images, labels)

modelNN.neighbours=images;
modelNN.labels=labels;

gaborFeaturesTrain = arrayfun(@(i) gabor_feature_vector(reshape(images(i, :), [27, 18])), 1:size(images, 1), 'UniformOutput', false);
trainFeatures = cell2mat(gaborFeaturesTrain');

modelNN.gabor_feature=trainFeatures;


end