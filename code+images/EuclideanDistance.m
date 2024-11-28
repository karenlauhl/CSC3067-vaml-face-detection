function dEuc = EuclideanDistance(sample1, sample2)
    % Convert to column vectors in case they are not
    sample1 = sample1(:);
    sample2 = sample2(:);
    
    % Compute the Euclidean distance
    dEuc = sqrt(sum((sample1 - sample2).^2));
end
