function Svm = svm(X, y, positiveClass, parameter)
tic

y = cellfun(@(y) char(y), cellstr(string(y)), 'UniformOutput',false);
uy = unique(y);
if numel(uy) ~= 2,     error("Input data has to be from 2 classes");    end
if ~isempty(positiveClass)
    positiveClass = cellstr(char(string(positiveClass)));
else
    positiveClass = y(1); % take the first label of y as positive class if not provided.
end
PosNegClass = [positiveClass; setdiff(uy, positiveClass)];
[~,y] = ismember(y, PosNegClass);   
y = logical(2 - y);     % tokenize labels to 0 (negative class) and 1 (positive class)

parameter = setDefaultSVMParameter(parameter);

[N, M] = size(X);
metadata.N = N;
metadata.M = M;

Svm = allocateSVM(metadata, parameter);
Svm = train_svm_cv(Svm, X, y, metadata, parameter);

Svm.PosNegClass = PosNegClass;
Svm.runtime = toc;
end

%% 

function Svm = train_svm_cv(Svm, X, y, metadata, parameter)
IterationLimit = 125 * metadata.N;

n_metrics = numel(parameter.metrics);
train_perf = nan(parameter.g,n_metrics);    % training test set performance evaluations
vali_perf = nan(parameter.g,n_metrics);

rng(parameter.seeds(1));
indices = crossvalind('Kfold', metadata.N, parameter.g);
for j = 1:parameter.g
    idx_vali = indices == j;
    idx_train = ~ idx_vali;
    model_cr = gather(fitcsvm(gpuArray(X(idx_train,:)),y(idx_train),'KernelFunction','linear',"Verbose",1,"IterationLimit",4/5*IterationLimit,"Solver","SMO","Alpha",0.0*ones(sum(idx_train),1)));
    v(j,:) = [model_cr.Beta',model_cr.Bias];

    Q = X(idx_train,:)* v(j,1:end-1)' +  v(end);
    y_pred = zeros(sum(idx_train), 1, 'logical'); 
    y_pred((Q>=0)) = 1; 
    cm = computeCM(y(idx_train), y_pred); % compute confusion matrix 
    for k=1:n_metrics
        train_perf(j,k) = evaluate_metrics([],[],cm.tp,cm.tn,cm.fp,cm.fn, parameter.metrics(k), parameter.metrics_predisposition(k)); % get training test set performance evaluations, considering equally positive and negative class.
    end
    
    Q = X(idx_vali,:)* v(j,1:end-1)' +  v(end);
    y_pred = zeros(sum(idx_vali), 1, 'logical'); 
    y_pred((Q>=0)) = 1; 
    cm = computeCM(y(idx_vali), y_pred); % compute confusion matrix 
    for k=1:n_metrics
        vali_perf(j,k) = evaluate_metrics([],[],cm.tp,cm.tn,cm.fp,cm.fn, parameter.metrics(k), parameter.metrics_predisposition(k)); % get training test set performance evaluations, considering equally positive and negative class.
    end
end
Svm.train_perf = train_perf;
Svm.vali_perf = vali_perf;
Svm.mean_train_perf = mean(train_perf,1,'omitnan'); % average performance evaluations on train set across all cross-validations
Svm.mean_vali_perf = mean(vali_perf,1,'omitnan'); % average performance evaluations on training test set across all cross-validations
Svm.v = mean(v,1,"omitmissing"); 
end


function cm = computeCM(y,y_pred)
cm = allocateCM();
cm.tp = sum(y_pred == 1 & y == 1);
cm.tn = sum(y_pred == 0 & y == 0);
cm.fp = sum(y_pred == 1 & y == 0);
cm.fn = sum(y_pred == 0 & y == 1);
end


%% Memory preallocation

function cm = allocateCM()
cm.tp = zeros(1,"uint64");
cm.tn = zeros(1,"uint64");
cm.fp = zeros(1,"uint64");
cm.fn = zeros(1,"uint64");
end


function Svm = allocateSVM(metadata, parameter)     
Svm.v = nan(1, metadata.M + 1);
Svm.train_perf = nan(parameter.g, length(parameter.metrics));
Svm.vali_perf = nan(parameter.g, length(parameter.metrics));
Svm.mean_train_perf = nan(1, length(parameter.metrics));
Svm.mean_vali_perf = nan(1, length(parameter.metrics));
Svm.test_perf = nan(1, length(parameter.metrics));
Svm.parameter = parameter;
Svm.PosNegClass = cell(2,1);
end


%% Set SVM Default Parameters
function parameter = setDefaultSVMParameter(parameter)
if ~isfield(parameter, "metrics"),                      parameter.metrics=["AUROC","AUPR","Fscore","ACscore","acc"]; end
if ~isfield(parameter, "metrics_predisposition"),       parameter.metrics_predisposition=["average","average","average","average","average"]; end
if ~isfield(parameter, "g"),                            parameter.g = 5;                end
if ~isfield(parameter, "parallel"),                     parameter.parallel=0;           end
if ~isfield(parameter, "seeds"),                        parameter.seeds=cell2mat(arrayfun(@(x)x.seeds,load('data/cr_seeds.mat','seeds'),'UniformOutput',false)); end
end
