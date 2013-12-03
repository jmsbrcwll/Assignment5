%% Setup variables
ziptrain = importdata('data/ziptrain.mat'); % ziptrain

train_target = ziptrain(:,1);
train_input = ziptrain(:,2:end);
%train_input = ziptrain(1:1000,2:end);

n_train = size(train_input,1);      % L
n_dimensions = size(train_input,2); % n
n_epochs = 1; % m

%% Algorithm
% Initialisation
max_digit = 9;
GLBcls = zeros(max_digit+1,n_train); % 1 = t
a = zeros(max_digit+1,n_train);
poly_degree = 3;

K_matrix_train = zeros(n_train,n_train);
for i = 1:n_train,
    for j = i:n_train,
        K_matrix_train(i,j) = K_fcn(train_input(i,:),train_input(j,:),poly_degree);
    end
end
K_matrix_train = K_matrix_train + (K_matrix_train' .* (ones(n_train)-eye(n_train)*0.5));

% Train
tth_instance = 0;
for loop = 1:1, % until convergence
    for i = 1:n_train,
        val = train_target(i,1); % Target
        preds = zeros(1,10);
        for j = 1:10, % Digit
            sum = 0;
            for z = 1:n_train,
                sum = sum + GLBcls(j,z)*K_matrix_train(i,z);
            end
            preds(j) = sum;
        end
        
        maxc = -inf;
        for j = 1:10,
            if (val == j),
                y = 1;
            else
                y = -1;
            end
            
            if (y*preds(j) <= 0),
                GLBcls(j,i) = GLBcls(j,i) - mysign(preds(j));
            end
            if (preds(j) > maxc),
               maxc = preds(j);
               maxi = j;
            end
                
        end
        
    end
end

% Validation (on training data)
validation_input = train_input;
validation_target = train_target;
n_validation = size(validation_input,1);   

K_matrix_validation = zeros(n_validation,n_validation);
for i = 1:n_validation,
    for j = 1:n_validation,
        K_matrix_validation(i,j) = K_fcn(validation_input(i,:),validation_input(j,:),poly_degree);
    end
end
correct = 0;

for i = 1:n_validation,
    val = train_target(i,1); % Target
    preds = zeros(1,10);
    for j = 1:10, % Digit
        sum = 0;
        for z = 1:n_validation,
            sum = sum + GLBcls(j,z)*K_matrix_validation(i,z);
        end
        preds(j) = sum;
    end

    maxc = -inf;
    for j = 1:10,
        if (val == j),
            y = 1;
        else
            y = -1;
        end
        
        if (preds(j) > maxc),
           maxc = preds(j);
           maxi = j;
        end
    end
    if maxi == val,
        correct = correct + 1;
    end
end
accuracy = correct / n_train
