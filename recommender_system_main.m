clc
clear all
close all

tic;
A_mat = xlsread('ratings.xlsx'); %A_mat is the given ratings matrix
oldRatingMatrix = A_mat;
R = (A_mat ~= 0); %R is the indicator matrix where 1 indicates the restaurant is rated and 0 indicates not rated

A_mat = A_mat.'; R = R.';
Y = A_mat;  % Normalization not required since the ratings are on the same scale

%% Iterative optimization - Collaborative Filtering

num_users = size(Y, 2);
num_restaurants = size(Y, 1);
num_features = 10;

% Randomly initialize Theta and X and update them iteratively
X = randn(num_restaurants, num_features);
Theta = randn(num_users, num_features);

initial_parameters = [X(:); Theta(:)];

% Set options for fmincg optimization
options = optimset('GradObj', 'on', 'MaxIter', 500);

% Regularization to avoid machine from blowing up

% lambda_vec = [0.001 0.01 0.1 0.5 1 5 10 20];  %Code to pick lambda
% for lambda_ctr = 1:length(lambda_vec)
%     lambda = lambda_vec(lambda_ctr);

lambda = 0.1; %Lambda was picked by checking the machine error values
theta = fmincg (@(t)(CostFunction(t, Y, R, num_users, num_restaurants, ...
                                num_features, lambda)), ...
                initial_parameters, options);

% Reverse fmincg effects
X = reshape(theta(1:num_restaurants*num_features), num_restaurants, num_features);
Theta = reshape(theta(num_restaurants*num_features+1:end), ...
                num_users, num_features);

fprintf('Model Learning is done.\n');

%% Prediction using the learned model

prediction = X * Theta';
prediction = round(prediction);

err_mat = (prediction.*R - Y.*R).^2; %Frobenius norm to find error between predicted rating matrix and given rating matrix only where ratings were provided
machine_err = mean(err_mat(:));
% machine_err_vec(lambda_ctr) = machine_err;
% end
% figure();
% semilogx(lambda_vec, machine_err_vec, '-xb','LineWidth',2, 'MarkerSize', 3);
% title('Justification of the choice of Lambda - regularization parameter');
% xlabel('Lambda');
% ylabel('Machine Error');
% grid on;

%Clipping the boundary values to meaningful range
prediction((prediction<=0))=1;
prediction((prediction>=5))=5;

Y_new = Y;
Y_new(R==0) = prediction(R==0); 

newRatingMatrix = Y_new.'; %newRatingMatrix is the new complete ratings matrix
%% Taking Distance feature into account

Dist_mat = xlsread('restaurants.xlsx');

% Compute Distance of all restaurants from you (0,0)
Dist_metric = [];
for i = 1:size(Dist_mat,1)
   temp = sqrt(Dist_mat(i,1).^2 + Dist_mat(i,2).^2);
   Dist_metric = [Dist_metric; temp];
end

my_ratings = Y_new(:,1);

%Normalize the metrics
my_ratings_norm = my_ratings/sum(my_ratings);
Dist_metric_inv = 1./(Dist_metric);
Dist_metric_inv_norm = Dist_metric_inv/sum(Dist_metric_inv);

para = 0.5; %User defined parameter which denotes if distance or ratings are to given more weightage. Here equal weightage has been chosen
score = para*my_ratings_norm + (1-para)*Dist_metric_inv_norm; %The score is to be maximized
score = score/sum(score); %Normalized Score

%% Use inverse transform sampling to arrive at the required distribution (Uses transformation of random variables)

IntVec = 1:length(score); 
cdf = cumsum(score); 

samples=1000; 
for j=1:samples 
    X_RV(j) = min(IntVec(cdf>=rand));
end

for i = 1 : length(score) 
    P(i) = sum(X_RV==i)/samples; 
end 

Rest_count = P*samples; %Shows how many times each restaurant has been visited
elapsed_time = toc;
%% Plotting the restaurant-visit frequency distributions

Restaur_list = {{'Burger', 'Palace'},{'Pizza',' Rica'},{'China',' Kitchen'},{'Tacos4Less'},{'Night',' Brunch'},...
    {'BBQ',' Outlet'},{'Steak',' Shanty'},{'Seafood',' OMalleys'},{'Mambo',' Grille'},{'El Veganio'},{'Hipsters',' Only'}...
    ,{'Doc Cajuns',' Shrimperia'},{'Just',' Mushrooms'},{'Ghana-to-g'},{'Derelicte'}};

figure();
vecX = IntVec-1; vecY = Rest_count;
plot(vecX,vecY,'-xb','LineWidth',2,'MarkerEdgeColor', 'r', 'MarkerSize', 8);
hold on;
for i = 1:length(IntVec)
        if(length(Restaur_list{i})==1)
        str = {Restaur_list{i}{1},strcat('Ratings:', num2str(my_ratings(i))), strcat('Dist:', num2str(Dist_metric(i)))};
        else
        str = {Restaur_list{i}{1},Restaur_list{i}{2},strcat('Ratings:', num2str(my_ratings(i))), strcat('Dist:', num2str(Dist_metric(i)))};
        end
    text(vecX(i)+0.37, vecY(i) - 5, (str),...
                'horizontalalignment','center','verticalalignment','bottom');
end
grid off;
title('Frequency Distribuitions of the Restaurant Visits (Line plot) - 1000 random trials ');
ylabel('Count');
hold off;

figure();
hist(X_RV,length(score))
[n,x] = hist(X_RV,length(score));
for i = 1:length(IntVec)
        if(length(Restaur_list{i})==1)
        str = {Restaur_list{i}{1},strcat('Ratings:', num2str(my_ratings(i))), strcat('Dist:', num2str(Dist_metric(i)))};
        else
        str = {Restaur_list{i}{1},Restaur_list{i}{2},strcat('Ratings:', num2str(my_ratings(i))), strcat('Dist:', num2str(Dist_metric(i)))};
        end
    text(x(i), n(i), (str),...
      'horizontalalignment','center','verticalalignment','bottom');
end
title('Frequency Distribuitions of the Restaurant Visits (Bar plot) - 1000 random trials ');
ylabel('Count');
grid off;

%% Display meaningful data

display(num_users);
display(num_restaurants);
display(num_features);
display(lambda);
display(oldRatingMatrix);
display(newRatingMatrix);
display(machine_err);
display(my_ratings);
display(Dist_metric);
display(score);
display(elapsed_time);



