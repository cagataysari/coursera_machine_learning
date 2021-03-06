function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%%%%%%%%%%%%%%%%%%% Start of Part 1 

%%% forward pass
X = [ones(m, 1) X]; % add bias to X
a1 = X;
z2 = a1*Theta1'; % compute a1
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3); 
%%%%
 

first = 0;
second = 0;

y_vect = zeros(1,size(a3,2));
for i = 1:m
y_vect(y(i))=1;
first += -y_vect*log(a3(i,:)');
second += ((1-y_vect)*log(1-a3(i,:)'));
y_vect(y(i)) = 0;
end
J = (1/m)*(first-second);

Theta1_r = Theta1(:,2:end).^2; % don't regularize the x0's weight
Theta2_r = Theta2(:,2:end).^2; % don't regularize the a20's weight

Thetas = (lambda/(2*m))*(sum(Theta1_r(:))+sum(Theta2_r(:)));

J += Thetas;
 
for k = 1:m
	
	y_vect(y(k))=1;
	a_back_1 = X(k,:);
	z_back_2 = a_back_1*Theta1';
	a_back_2 = sigmoid(z_back_2);
	a_back_2 = [ones(size(a_back_2,1),1) a_back_2];
	z_back_3 = a_back_2*Theta2';
	a_back_3 = sigmoid(z_back_3);
	
	delta_3 = (a_back_3-y_vect);
	delta_2 = (delta_3*Theta2);
	delta_2 = delta_2(:,2:end); 
	delta_2 = delta_2.*sigmoidGradient(z_back_2);
	
	Theta2_grad += (1/m)*(delta_3'*a_back_2);
	Theta1_grad += (1/m)*(delta_2'*a_back_1);
    y_vect(y(k)) = 0;
end

 
Theta1_grad(:,2:end) += (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) += (lambda/m)*Theta2(:,2:end);

%%%%%%%%%%%%%%%%%%% End of Part 1 

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
