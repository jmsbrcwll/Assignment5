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
w = zeros(100,10); % 1 = t
a = zeros(10,n_train);
K_matrix = zeros(n_train,n_train);
max_digit = 9;
poly_degree = 3;


for i = 1:n_train,
    for j = 1:n_train,
        K_matrix(i,j) = K_fcn(train_input(i,:),train_input(j,:),3);
    end
end

% Binarise targets
for target_digit = 0:max_digit,
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
    tth_instance = 0;
    for loop = 1:1, % until convergence
        for t = 1:n_train,
            y_target = gen_targets(t);
            
            % Prediction
            y_predict = mysign(w_fcn(w(:,index),K_matrix,t));
            
            if y_predict ~= y_target,
                a(index,t) = y_predict;
            else
                a(index,t) = 0;
            end

           %w(tth_instance+1,index) = dot(train_input(index,t),train_input(index,t))^poly_degree; % 100 x 10
            %w(tth_instance+1,index+max_digit+1) = a(index,t);
            w(tth_instance+1,index) = a(index,t);
            tth_instance = tth_instance + 1;
        end
    end
end

% Test

% prediction = [];
% for i = 1:n_train,
%     current_prediction_vec = [];
%     for j = 1:10,
%          sum = 0;
%          for k = 1:length(w),
%              sum = sum + a(j,k)*dot(train_input(k,:),train_input(i,:))^poly_degree;
%          end
%          current_prediction_vec(j,1) = sum;
%     end
%     prediction(i,1) = find(current_prediction_vec==max(current_prediction_vec))-1;
%     prediction(i,2) = train_target(i,1);
% end
% accuracy = (nnz(find(prediction(:,1)==prediction(:,2)))/n_train)
