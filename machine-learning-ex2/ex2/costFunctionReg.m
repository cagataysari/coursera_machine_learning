function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


J_first = costFunction(theta,X,y);
thetas = 0;
n = length(theta);

for iter = 2:n
	thetas += (theta(iter)**2);
end

thetas = (lambda/(2*m))*thetas;
J = J_first+thetas;
grad = zeros(size(theta));

k = X*theta;
h_X = sigmoid(k);

grad(1) = (1/m)*sum((h_X-y).*X(:,1));

for iter = 2:n
	grad(iter) = (1/m)*sum((h_X-y).*X(:,iter)) + ((lambda/m)*theta(iter));
end 




% =============================================================

end
