function [J, grad] = CostFunction(params, Y, R, num_users, num_restaurants, ...
                                  num_features, lambda)
                              
%CostFunction returns the cost and gradient for the collaborative filtering problem.


X = reshape(params(1:num_restaurants*num_features), num_restaurants, num_features);
Theta = reshape(params(num_restaurants*num_features+1:end), ...
                num_users, num_features);

J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

errors = ((X * Theta' - Y) .* R);
squaredErrors = errors .^ 2;
J = ((1 / 2) * sum(squaredErrors(:))) + ((lambda / 2) * sum(Theta(:) .^ 2)) + ((lambda / 2) * sum(X(:) .^ 2));

X_grad = errors * Theta + (lambda .* X);
Theta_grad = errors' * X + (lambda .* Theta);

grad = [X_grad(:); Theta_grad(:)];

end
