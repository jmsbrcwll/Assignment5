%%
% To be ran after train_perceptron
%%

%% Setup variables 
% Test
ziptest = importdata('data/ziptest.mat');
test_target = ziptest(:,1);
test_input = ziptest(:,2:end);   
n_test = size(test_input,1);      

recompute = 0; % 0 = don't recompute Kmatrix, 1 = recompute Kmatrix

%% Algorithm
% Testing
fprintf('Testing...\n');
fprintf('  Loading Kmatrix (test)\n');
if ~exist('K_matrix_test','var') || recompute == 1,
    K_matrix_test = zeros(n_test,n_train);
    for i = 1:n_test,
        dat = repmat(test_input(i,:),[n_train,1]);
        K_matrix_test(i,:) = dot(dat',train_input');
    end
end
K_matrix_test_polyd = K_matrix_test .^ poly_degree;


correct = 0;
test_details_confidence = zeros(10,n_test);
for i = 1:n_test,
    val = test_target(i,1); % Target 0-9
    if val == 0, 
        val = 10;
    end
    preds = zeros(1,10);
    for j = 1:10, % Digit
        preds(j) = sum(GLBcls(j,:).*K_matrix_test_polyd(i,:)); 
        test_details_confidence(j,i) = preds(j);
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
    if maxi == 10,
        maxi = 0;
    end
    if val == 10,
        val = 0;
    end
    test_details(i,1) = maxi; % Predicted
    test_details(i,2) = val; % Actual
end
accuracy = correct / n_test;

fprintf('Completed\n');
fprintf('  Test Set Size: %d\n',n_train);
fprintf('  Poly Dimension: %d\n',poly_degree);
fprintf('  Accuracy: %f\n',accuracy);
fprintf('  Test Error: %f\n',1-accuracy);
