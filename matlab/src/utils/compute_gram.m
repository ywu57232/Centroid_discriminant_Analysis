function K = compute_gram(classifier, X1, X2)
% Compute the squared Euclidean distance matrix
sq_dist = pdist2((X1), (X2), 'squaredeuclidean');
% Compute the kernel matrix using vectorization
K = exp(-sq_dist / (2 * classifier.sigma^2));
end