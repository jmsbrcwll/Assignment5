%% Setup variables
ziptrain = importdata('data/ziptrain.mat'); % ziptrain

train_target = ziptrain(:,1);
train_input = ziptrain(1:5000,2:end);
recompute = 0; % 0 = don't recompute Kmatrix, 1 = recompute Kmatrix

n_train = size(train_input,1);      
n_epochs = 20; % Hard maximum on the number of epochs

%% Algorithm
% Initialisation
max_digit = 9;
GLBcls = zeros(max_digit+1,n_train); % 1 = t
previous_assignment = ones(max_digit+1,n_train).*-1;
poly_degree = 4;
convergedEpoch = -1;

% Train
fprintf('Training...\n');
fprintf('  Loading Kmatrix (train)\n');
if ~exist('K_matrix_train','var') || recompute == 1,
    K_matrix_train = zeros(n_train,n_train);
    for i = 1:n_train,
        dat = repmat(train_input(i,:),[n_train,1]);
        K_matrix_train(i,:) = dot(dat',train_input');
    end
end
K_matrix_train_polyd = K_matrix_train .^ poly_degree;

fprintf('  Epoch:');
for loop = 1:n_epochs, % until convergence
    if (previous_assignment == GLBcls),
        convergedEpoch = loop-1;
        fprintf('\n  Converged');
        break;
    else
        previous_assignment = GLBcls;
    end
    fprintf(' %d',loop);
    for i = 1:n_train,
        val = train_target(i,1); % Target
        if val == 0, 
            val = 10;
        end
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

fprintf('\nTraining Complete\nValidating...\n');
% Validation
validation_input = ziptrain(5001:end,2:end);
validation_target = ziptrain(5001:end,1);
n_validation = size(validation_input,1);   

fprintf('  Loading Kmatrix (validation)\n');
if ~exist('K_matrix_validation','var') || recompute == 1,
    K_matrix_validation = zeros(n_validation,n_train);
    for i = 1:n_validation,
        dat = repmat(validation_input(i,:),[n_train,1]);
        K_matrix_validation(i,:) = dot(dat',train_input');
    end
end
K_matrix_validation_polyd = K_matrix_validation .^ poly_degree;

correct = 0;
for i = 1:n_validation,
    val = validation_target(i,1); % Target 0-9
    if val == 0, 
        val = 10;
    end
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
        dat = repmat(train_input(i,:),[n_train,1]);
        K_matrix_train_all(i,:) = dot(dat',train_input');
    end
end
K_matrix_train_all_polyd = K_matrix_train_all .^ poly_degree;

% Train
fprintf('Retraining...\n');
fprintf('  Epoch:');
for loop = 1:n_epochs, % until convergence
    if (previous_assignment == GLBcls),
        convergedEpoch = loop-1;
        fprintf('\n  Converged\n');
        break;
    else
        previous_assignment = GLBcls;
    end
    fprintf(' %d',loop);
    for i = 1:n_train,
        val = train_target(i,1); % Target
        if val == 0, 
            val = 10;
        end
        
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
