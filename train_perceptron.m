%% Setup variables
ziptrain = importdata('data/ziptrain.mat'); % ziptrain

train_target = ziptrain(:,1);
train_input = ziptrain(1:5000,2:end);
recompute = 0; % 0 = don't recompute Kmatrix, 1 = recompute Kmatrix

n_train = size(train_input,1);      
n_epochs = 100; % m

%% Algorithm
% Initialisation
max_digit = 9;
GLBcls = zeros(max_digit+1,n_train); % 1 = t
previous_assignment = ones(max_digit+1,n_train).*-1;
poly_degree = 3;
convergedEpoch = -1;

% Train
fprintf('Training...\n');
fprintf('  Loading Kmatrix (train)\n');
if ~exist('K_matrix_train','var') || recompute == 1,
    K_matrix_train = zeros(n_train,n_train);
    for i = 1:n_train,
        for j = i:n_train,
            K_matrix_train(i,j) = K_fcn(train_input(i,:),train_input(j,:),1);
        end
    end
    % Append transpose and half the diagonal as it was doubled in our addition
    % Cuts number of computations required in half
    K_matrix_train = K_matrix_train + (K_matrix_train' .* (ones(n_train)-eye(n_train)*0.5));
end
 K_matrix_train_polyd =exp(-poly_degree.* K_matrix_train);

for loop = 1:n_epochs, % until convergence
    if (previous_assignment == GLBcls),
        convergedEpoch = loop-1;
        fprintf('  Converged\n');
        break;
    else
        previous_assignment = GLBcls;
    end
    fprintf('  Epoch %d\n',loop);
    for i = 1:n_train,
        val = train_target(i,1); % Target
        preds = zeros(1,10);
        for j = 1:10, % Digit
            preds(j) = sum(GLBcls(j,:).*K_matrix_train_polyd(i,:));
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

fprintf('Training Complete\nValidating...\n');
% Validation
validation_input = ziptrain(5001:end,2:end);
validation_target = ziptrain(5001:end,1);
n_validation = size(validation_input,1);   

fprintf('  Loading Kmatrix (validation)\n');
if ~exist('K_matrix_validation','var') || recompute == 1,
    K_matrix_validation = zeros(n_validation,n_train);
    for i = 1:n_validation,
        for j = 1:n_train,
            K_matrix_validation(i,j) = K_fcn(validation_input(i,:),train_input(j,:),1);
        end
    end
end
K_matrix_validation_polyd =exp(-poly_degree.* K_matrix_validation);


correct = 0;
for i = 1:n_validation,
    val = validation_target(i,1); % Target 0-9
    preds = zeros(1,10);
    for j = 1:10, % Digit
        preds(j) = sum(GLBcls(j,:).*K_matrix_validation_polyd(i,:));        
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
    if maxi == 10,
        maxi = 0;
    end
    if maxi == val,
        correct = correct + 1;
    end
end
accuracy = correct / n_validation;

fprintf('Completed\n');
fprintf('  Training Set Size: %d\n',n_train);
fprintf('  Validation Set Size: %d\n',n_validation);
fprintf('  Poly Dimension: %d\n',poly_degree);
fprintf('  Epochs: %d\n',convergedEpoch);
fprintf('  Hold-out Accuracy: %f\n',accuracy);
fprintf('  Hold-out Error: %f\n',1-accuracy);

fprintf('Retraining on all data (T+V)\n');
train_input = ziptrain(:,2:end);
n_train = size(train_input,1);      

GLBcls = zeros(max_digit+1,n_train); 
previous_assignment = ones(max_digit+1,n_train).*-1;
convergedEpoch = -1;

if ~exist('K_matrix_train_all','var') || recompute == 1,
    K_matrix_train_all = zeros(n_train,n_train);
    for i = 1:n_train,
        for j = i:n_train,
            K_matrix_train_all(i,j) = Gaussian_K_fcn(train_input(i,:),train_input(j,:),1);
        end
    end
    K_matrix_train_all = K_matrix_train_all + (K_matrix_train_all' .* (ones(n_train)-eye(n_train)*0.5));
end

K_matrix_train_all_polyd =exp(-poly_degree.* K_matrix_train_all);

% Train
fprintf('Retraining...\n');
for loop = 1:n_epochs, % until convergence
    if (previous_assignment == GLBcls),
        convergedEpoch = loop-1;
        fprintf('  Converged\n');
        break;
    else
        previous_assignment = GLBcls;
    end
    fprintf('  Epoch %d\n',loop);
    for i = 1:n_train,
        val = train_target(i,1); % Target
        preds = zeros(1,10);
        for j = 1:10, % Digit
            preds(j) = sum(GLBcls(j,:).*K_matrix_train_all_polyd(i,:));
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

fprintf('Retraining Complete\n');

test_perceptron;
