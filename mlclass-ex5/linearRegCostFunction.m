function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

predictions = X * theta;

sqrErrors = (predictions - y) .^2;

sum = sum(sqrErrors(:));


theta2 = theta;

theta2(1) = 0;

penalty = (lambda / (2 * m)) * (theta2' * theta2);

J = (1 / (2 * m)) * sum + penalty;


gradOld = X' * (predictions .- y) ./ m;

grad = gradOld .+ (lambda / m) .* theta2;




% =========================================================================

grad = grad(:);

end
