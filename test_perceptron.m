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
        for j = 1:n_train,
            K_matrix_test(i,j) = Gaussian_K_fcn(test_input(i,:),train_input(j,:),1);
        end
    end
end

K_matrix_test_polyd =exp(-poly_degree.* K_matrix_test);

correct = 0;
for i = 1:n_test,
    val = test_target(i,1); % Target 0-9
    preds = zeros(1,10);
    for j = 1:10, % Digit
        preds(j) = sum(GLBcls(j,:).*K_matrix_test_polyd(i,:));        
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
accuracy = correct / n_test;

fprintf('Completed\n');
fprintf('  Test Set Size: %d\n',n_train);
fprintf('  Poly Dimension: %d\n',poly_degree);
fprintf('  Accuracy: %f\n',accuracy);
fprintf('  Test Error: %f\n',1-accuracy);
