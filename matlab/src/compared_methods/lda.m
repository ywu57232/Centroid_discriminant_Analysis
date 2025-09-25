function Lda = lda(X, y, positiveClass, parameter)
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

parameter = setDefaultLdaParameter(parameter);

[N, M] = size(X);
metadata.N = N;
metadata.M = M;

Lda = allocateLda(metadata, parameter);
Lda = train_svm_cv(Lda, X, y, metadata, parameter);

Lda.PosNegClass = PosNegClass;
Lda.runtime = toc;
end

%% 

function Lda = train_svm_cv(Lda, X, y, metadata, parameter)
IterationLimit = 125 * metadata.N;

n_metrics = numel(parameter.metrics);
train_perf = nan(parameter.g,n_metrics);    % training test set performance evaluations
vali_perf = nan(parameter.g,n_metrics);

rng(parameter.seeds(1));
indices = crossvalind('Kfold', metadata.N, parameter.g);
for j = 1:parameter.g
    idx_vali = indices == j;
    idx_train = ~ idx_vali;
    model_cr = fitcdiscr(X(idx_train,:),y(idx_train),'discrimType','pseudolinear','OptimizeHyperparameters','none');
    v(j,:) = [model_cr.Coeffs(2,1).Linear;model_cr.Coeffs(2,1).Const];

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
Lda.train_perf = train_perf;
Lda.vali_perf = vali_perf;
Lda.mean_train_perf = mean(train_perf,1,'omitnan'); % average performance evaluations on train set across all cross-validations
Lda.mean_vali_perf = mean(vali_perf,1,'omitnan'); % average performance evaluations on training test set across all cross-validations
Lda.v = mean(v,1,"omitmissing"); 
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


function Lda = allocateLda(metadata, parameter)     % basic centroid-vector struct
Lda.v = nan(1, metadata.M + 1);
Lda.train_perf = nan(parameter.g, length(parameter.metrics));
Lda.vali_perf = nan(parameter.g, length(parameter.metrics));
Lda.mean_train_perf = nan(1, length(parameter.metrics));
Lda.mean_vali_perf = nan(1, length(parameter.metrics));
Lda.test_perf = nan(1, length(parameter.metrics));
Lda.parameter = parameter;
Lda.PosNegClass = cell(2,1);
end


%% Set CDA Default Parameters
function parameter = setDefaultLdaParameter(parameter)
if ~isfield(parameter, "metrics"),                      parameter.metrics=["AUROC","AUPR","Fscore","ACscore","acc"]; end
if ~isfield(parameter, "metrics_predisposition"),       parameter.metrics_predisposition=["average","average","average","average","average"]; end
if ~isfield(parameter, "g"),                            parameter.g = 5;                end
if ~isfield(parameter, "parallel"),                     parameter.parallel=0;           end
if ~isfield(parameter, "seeds"),                        parameter.seeds=cell2mat(arrayfun(@(x)x.seeds,load('data/cr_seeds.mat','seeds'),'UniformOutput',false)); end
end
