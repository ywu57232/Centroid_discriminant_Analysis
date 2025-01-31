function [trained, multipredicted, parameter] = train_dataset(X, y, XtoPred, trained, classifier, parameter)
% A multiclass training and predicting framework for Centroid Discriminant Analysis (CDA), as well
% as for other binary classifiers.

% X: Multiclass samples to classify in a 2d NxM matrix. Rows are N observations and columns are
% M features.
% y: Multiclass sample labels in a Nx1 array, in string cell or numeric array. 

% classifier: Binary-class classifiers to be tested on the data, in a struct. 

% parameter: 
% parameter.train_mode: "train_pairs": train models from data;  "train_and_multipredict": train binary models and perform multiclass prediction; "multiclass_predict": predict data using trained models.
% parameter.data_mode: "train-test"(default, 4 to 1 fold): split to train and test set, "train_all_data": train all input data.

y_info = whos('y');
y = cellfun(@(y) char(y), cellstr(string(y)), 'UniformOutput',false); % convert labels to character cells

parameter = setDefaultParameter(parameter);

if isempty(parameter.data_mode), parameter.data_mode = "train-test"; end

metadata = get_metadata(X, y, [], parameter);

clf_name = fieldnames(classifier);
for k = 1:length(clf_name)
    if ~isfield(classifier.(clf_name{k}), "parameter")
        classifier.(clf_name{k}).parameter = [];
    end
    classifier.(clf_name{k}).name = clf_name{k};

    % Train binary-pair models
    if isempty(trained)
        trained.(clf_name{k}).model = train_and_test(X, y, metadata, classifier.(clf_name{k}), parameter);
    
        trained.(clf_name{k}).parameter = trained.(clf_name{k}).model(1).parameter;
        trained.(clf_name{k}).model = rmfield(trained.(clf_name{k}).model, "parameter");
    end

    % Predict multicass using trained binary-pair models via ECOC
    multipredicted = [];
    if strcmp(parameter.train_mode, 'train_and_multipredict')
        test_indices = cell2mat(cellfun(@(x, y) y(x == 1), metadata.indices, metadata.classLoc, 'UniformOutput', false)');
        multipredicted.(clf_name{k}) = multipredict(X(test_indices,:), y(test_indices), metadata, trained.(clf_name{k}), clf_name{k}, parameter);
    elseif strcmp(parameter.train_mode, 'multiclass-predict')
        multipredicted.(clf_name{k}) = multipredict(XtoPred, [], metadata, trained.(clf_name{k}), clf_name{k}, parameter);
        if ~ strcmp(y_info.class, 'cell')
            multipredicted.(clf_name{k}).y_pred = [cellfun(@(x) eval([y_info.class,'(str2double(x))']), multipredicted.(clf_name{k}).y_pred)];
        end
    end    
end
trained.parameter = parameter;
end


function trained = train_and_test(X, y, metadata, classifier, parameter)
k=1;
disp('Start training ...')
for i = 1:metadata.numClass - 1
    for j = i + 1 : metadata.numClass
        [train, test] = create_binary_pair(X, y, metadata, i ,j, parameter);
        trained(k) = eval([char(classifier.src),'(train.X, train.y, metadata.uy(i), classifier.parameter);']);
        if ~strcmp(parameter.data_mode, 'train_all_data')
            trained(k) = predict_from_model(test.X, test.y, classifier, trained(k), trained(k).parameter);
        end
        disp(['Trained ', num2str(k), ' of ', num2str(metadata.numClass * (metadata.numClass - 1) / 2), ' binary-pairs.'])
        k = k + 1;
    end
end
disp('Finish training.')
end


function trained = predict_from_model(X, y, classifier, trained, parameter)
y = cellfun(@(y) char(y), cellstr(string(y)), 'UniformOutput',false);
[~, y_tok] = ismember(y, trained.PosNegClass);
y_tok = logical(2 - y_tok); % tokenize to 0 and 1 (pos).
if strcmp(classifier.name,"cda")
    Q = X * trained.v';
    y_pred_tok = zeros(length(y), 1, 'logical'); 
    y_pred_tok((Q >= trained.oop)) = 1; % predict as positive to the right of OOP, negative to the left of OOP        
end
cm = computeCM(y_tok, y_pred_tok);  % compute confusion matrix       
for k = 1:numel(parameter.metrics)
    trained.test_perf(k) = evaluate_metrics([], [],cm.tp,cm.tn,cm.fp,cm.fn, parameter.metrics(k), parameter.metrics_predisposition(k)); % evaluate performance on each metric
end
end


function [train, test] = create_binary_pair(X, y, metadata, i, j, parameter)
if strcmp(parameter.data_mode, 'train-test')
    test_indices = [metadata.classLoc{i}(metadata.indices{i}==1);metadata.classLoc{j}(metadata.indices{j}==1)];
elseif strcmp(parameter.data_mode, 'train_all_data')
    test_indices = [];
end
train_indices = setdiff(union(metadata.classLoc{i},metadata.classLoc{j}),test_indices);
train.X = X(train_indices,:);
train.y = y(train_indices);
test.X = X(test_indices,:);    
test.y = y(test_indices);
end


function multipredicted = multipredict(X, y, metadata, classifier, clf_name, parameter)
Q = nan(size(X, 1), length(classifier.model));
for k = 1:length(classifier.model)
    switch clf_name
        case "cda"
            Q(:, k) = X * classifier.model(k).v' - classifier.model(k).oop;
    end
end

CodingMatrix = zeros(metadata.numClass, length(classifier.model));
k=1;
for i=1:metadata.numClass
    for j=i+1:metadata.numClass
        CodingMatrix(i,k) = (-1)^(1 + strcmp(metadata.uy{i}, classifier.model(k).PosNegClass{1}));
        CodingMatrix(j,k) = (-1)^(1 + strcmp(metadata.uy{j}, classifier.model(k).PosNegClass{1}));
        k=k+1;
    end
end

totalHingeLoss = nan(size(X, 1), metadata.numClass);
for C=1:metadata.numClass
    totalHingeLoss(:,C)=sum(1/2*max(0,0-CodingMatrix(C,:).*Q), 2);
end
[~,idx_min] = min(totalHingeLoss,[],2);
y_pred = metadata.uy(idx_min);

if strcmp(parameter.train_mode, 'multiclass-predict')
    multipredicted.y_pred = y_pred;
    return
end

CM = confusionmat(y, y_pred);
cm = repmat(struct('TP',0,'TN',0,'FP',0,'FN',0),metadata.numClass,1);
for C=1:metadata.numClass
    cm(C).tp = CM(C,C);
    cm(C).tn = sum(CM([1:C-1, C+1:end], [1:C-1, C+1:end]), "all");
    cm(C).fp = sum(CM([1:C-1, C+1:end], C));
    cm(C).fn = sum(CM(C, [1:C-1, C+1:end]));
    for k=1:numel(classifier.parameter.metrics)
        multipredicted.test_perf(C, k) = evaluate_metrics([],[],cm(C).tp,cm(C).tn,cm(C).fp,cm(C).fn, classifier.parameter.metrics(k), classifier.parameter.metrics_predisposition(k));
    end
end
multipredicted.mean_test_perf = mean(multipredicted.test_perf,"omitmissing");
end


function metadata = get_metadata(X, y, metadata, parameter)
metadata.uy = unique(y,'stable');
metadata.numClass = length(metadata.uy);
[metadata.N,metadata.M] = size(X);
metadata.numPerClass = cellfun(@(x) sum(ismember(y, x)), metadata.uy);
for i=1:metadata.numClass
    rng(parameter.assignSeed)
    parameter.assignSeed = parameter.assignSeed + 1;
    metadata.classLoc{i} = find(cellfun(@(x) strcmp(x, metadata.uy{i}), y));
    metadata.indices{i} = crossvalind('Kfold', metadata.numPerClass(i), 5);
end
end


function cm = computeCM(y,y_pred)
cm = allocateCM();
cm.tp = sum(y_pred == 1 & y == 1);
cm.tn = sum(y_pred == 0 & y == 0);
cm.fp = sum(y_pred == 1 & y == 0);
cm.fn = sum(y_pred == 0 & y == 1);
end

function cm = allocateCM()
cm.tp = zeros(1,"uint64");
cm.tn = zeros(1,"uint64");
cm.fp = zeros(1,"uint64");
cm.fn = zeros(1,"uint64");
end


%% Set Default Parameters
function parameter = setDefaultParameter(parameter)
if ~isfield(parameter, "train_mode"),                   parameter.train_mode = "train_pairs";           end
if ~isfield(parameter, "data_mode"),                    parameter.data_mode = "train-test";           end
if ~isfield(parameter, "trained_model"),                parameter.data_mode = [];           end
if ~isfield(parameter, "assignSeed"),                   parameter.assignSeed = 10086;           end
end
