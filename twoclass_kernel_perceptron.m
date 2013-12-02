%% Setup variables
ziptrain = importdata('data/ziptrain.mat'); % ziptrain

train_target = ziptrain(:,1);
train_input = ziptrain(:,2:end);
train_input = ziptrain(1:100,2:end);

n_train = size(train_input,1);      % L
n_dimensions = size(train_input,2); % n
n_epochs = 1; % m
d = 2; % positive integer controlling the dimension of the polynomial

%% Algorithm
% Initialisation
w = zeros(10,256);
a = zeros(10,n_train);
K_matrix = zeros(n_train,n_train);


for i = 1:n_train,
    for j = 1:n_train,
        K_matrix(i,j) = K_fcn(train_input(i,:),train_input(j,:),2);
    end
end

% Binarise targets
for target_digit = 0:9,
    index = target_digit + 1;
    
    gen_targets = [];
    for i = 1:n_train,
        if train_target(i,1) == target_digit,
            gen_targets(i) = 1;
        else
            gen_targets(i) = -1;
        end
    end

    % Train
    for loop = 1:5,
        for i = 1:n_train,
            y_target = gen_targets(i);
            y_predict = sign(sum(K_matrix(i,:) .* a(index,:) .* y_target) - 0.5);

            if y_predict ~= y_target,
                a(index,i) = y_predict;
            else
                a(index,i) = 0;
            end

            w(index,xxx) = w(index,xxx) + a(index,i) .* K_matrix(i,i);
        end
    end
end