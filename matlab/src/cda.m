function Cda = cda(X, y, positiveClass, parameter)
% Centroid Discriminant Analysis (CDA): A scalable classification algorithm
% for binary-class data. Multiclass prediction is supported via the ECOC framework using the trained binary-pair models.

% X: Binary-class samples to classify in a 2d NxM matrix. Rows are N observations and columns are
% M features.
% y: Binary-class sample labels in a Nx1 array. 
% classifier: Binary-class classifiers to be tested on the data, in a struct. 

% parameter: 
% parameter.train_mode: "train_pairs": train models from data, "multiclass_predict": predict from trained models.
% parameter.data_mode: "train-test"(default, 4 to 1 fold): split to train and test set, "train_all_data": train all input data.

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

parameter = setDefaultCdaParameter(parameter);

[N, M] = size(X);
metadata.N = N;
metadata.M = M;

alpha = ones(N,1) / sqrt(N);

Cda = allocateCda(metadata, parameter);
if ~strcmp(parameter.type, 'linear')
    dist_matrix = pdist2(X, X);
    Cda.sigma = sqrt(parameter.kernel_c) * median(dist_matrix(:));      % 0.1025, 0.1
end

Cda.Cdb1 = soft_centroid_vector(Cda.Cdb1, X, y, alpha, parameter);
if strcmp(parameter.type, 'linear')
    Cda.Cdb1.Q = X * Cda.Cdb1.v';
else
    Cda.Cdb1.Q = gather(compute_gram(Cda, X, X) * Cda.Cdb1.beta);
end
Cda.Cdb1 = train_oop_cv(Cda.Cdb1, y, parameter); 


iter = 1;
max_ps = 0;
bestCda = [];
ps_trace = nan(1,parameter.ps_trace_length);

while iter <= parameter.maxIter
    
    % update sample weights for soft-centroid computation
    alpha = update_sample_weight(alpha, X, y, Cda, parameter);

    % compute the soft-centroid vector CDB2; find the orthogonal vector to
    % CDB1
    Cda.Cdb2 = soft_centroid_vector(Cda.Cdb2, X, y, alpha, parameter);
    Cda.Cdb2.v_orth = Cda.Cdb2.v - Cda.Cdb1.v * Cda.Cdb2.v' * Cda.Cdb1.v;
    Cda.Cdb2.v_orth = Cda.Cdb2.v_orth / norm(Cda.Cdb2.v_orth);
    if strcmp(parameter.type, 'linear')
        Cda.Cdb2.v_orth = Cda.Cdb2.v_orth * sign(sign(Cda.Cdb2.v_orth(3)+0.1));
        Cda.Cdb2.Q_orth = X * Cda.Cdb2.v_orth';
    else
        norm_factor = norm(Cda.Cdb2.v_orth);
        Cda.Cdb2.v_orth = Cda.Cdb2.v_orth / norm_factor;
        Cda.Cdb2.beta_orth = (Cda.Cdb2.beta - Cda.Cdb1.v * Cda.Cdb2.v' * Cda.Cdb1.beta) / norm_factor * sign(sign(Cda.Cdb2.v_orth(3)+0.1));
        Cda.Cdb2.v_orth = Cda.Cdb2.v_orth * sign(sign(Cda.Cdb2.v_orth(3)+0.1));
        Cda.Cdb2.Q_orth = gather(compute_gram(Cda, X, X) * Cda.Cdb2.beta_orth);
    end

    % Bayesian optimization search for the approximated optimal vector on
    % the hyperplane spanned by CDB1 and CDB2.
    Cda = BO_search(X, y, Cda, min(10, iter + 3), iter, parameter);

    % record the best iteration
    ps_trace(1:parameter.ps_trace_length-1) = ps_trace(2:parameter.ps_trace_length);
    ps_trace(end) = Cda.ps;
    if Cda.ps > max_ps
        max_ps = Cda.ps;
        Cda.bestIter = iter;
        bestCda = Cda;    
    end
    bestCda.totalIter = iter;

    % check the coefficient of variance (CV) of performance-score (ps) to
    % stop the algorithm in advance
    if ~any(isnan(ps_trace))
        cv = std(ps_trace) / mean(ps_trace);
        if cv < parameter.cv_threshold      % 1e-3
            break
        end    
    end

    % let the CDA obtained at the end of each iteration be the new CDB1 of
    % the next iteration
    fields = fieldnames(Cda.Cdb1);
    for i = 1:length(fields)
        Cda.Cdb1.(fields{i}) = Cda.(fields{i});
    end

    iter = iter+1;
end
Cda = bestCda;

% check and refine CDA result by a statistical p-value examination
Cda = finalizeCda(X, y, Cda, metadata, parameter);

Cda.PosNegClass = PosNegClass;
Cda.runtime = toc;
Cda = rmfield(Cda,["Q","Cdb1","Cdb2"]);
end



function Cda = BO_search(X, y, Cda, BOIter, CDAIter, parameter)
fun = @(var_theta) lossfun(var_theta, y, Cda, parameter);
Mdl = BOImpl(fun, [-90,90], [], BOIter, CDAIter);
Cda.theta = Mdl.XAtMinEstimatedObjective;
Cda.v = Cda.Cdb1.v * cosd(Cda.theta) + Cda.Cdb2.v_orth * sind(Cda.theta);
if strcmp(parameter.type, 'linear')
    Cda.Q = X * Cda.v';
else
    Cda.beta = Cda.Cdb1.beta * cosd(Cda.theta) + Cda.Cdb2.beta_orth * sind(Cda.theta);
    Cda.Q = gather(compute_gram(Cda, X, X) * Cda.beta);
end
Cda = train_oop_cv(Cda, y, parameter); 
end


function loss = lossfun(var_theta, y, Cda, parameter)
Cda.Q = Cda.Cdb1.Q * cosd(var_theta) + Cda.Cdb2.Q_orth * sind(var_theta);
Cda = train_oop_cv(Cda, y, parameter); 
loss = - Cda.ps;
end


function Cdb = soft_centroid_vector(Cdb, X, y, alpha, parameter)
sign_map = [-1;1];
sum_of_alpha = [sum(alpha(y==0)); sum(alpha(y==1))];
if strcmp(parameter.type, 'linear')
    v = (sign_map(y+1) .* alpha ./ sum_of_alpha(y+1))' * X;
else
    beta = (sign_map(y+1) .* alpha ./ sum_of_alpha(y+1));
    v = beta' * X;
end
if ~ any(v)
    error('Cannot trace the vector from overlapped group centriods.');
end
normv = norm(v);
Cdb.v = v / normv;    % CDA eigenvector is the unit vector pointing from the centroid of negative class to that of positive class.
if ~strcmp(parameter.type, 'linear')
    Cdb.beta = beta / normv;
end
end


function alpha = update_sample_weight(alpha, X, y, Cda, parameter)
if strcmp(parameter.type, 'linear')
    Q = Cda.Cdb1.Q;
else
    Q = gather(compute_gram(Cda, X, X) * Cda.Cdb1.beta);
end
Cda.Cdb1 = train_oop_cv(Cda.Cdb1, y, parameter); 
d = abs(Q - Cda.Cdb1.oop);
d1 = d / sum(d);
d2 = abs(d1 - min(d1) - max(d1));
alpha = normalize(alpha .* d2, "norm");
end

%% 

function Cdb = train_oop_cv(Cdb, y, parameter)
N = length(y);

n_metrics = numel(parameter.metrics);
train_perf = nan(parameter.g,n_metrics);    % training test set performance evaluations
vali_perf = nan(parameter.g,n_metrics);
model_cr = struct('ps',cell(parameter.g,1),'oop',cell(parameter.g,1));

rng(parameter.seeds(1));
indices = crossvalind('Kfold', N, parameter.g);
for j = 1:parameter.g
    idx_vali = indices == j;
    idx_train = ~ idx_vali;
    model_cr(j) = train_oop(Cdb.Q(idx_train), y(idx_train), parameter); % find OOP for each cross-validation turn 
    
    y_pred = zeros(sum(idx_train), 1, 'logical'); 
    y_pred((Cdb.Q(idx_train) >= model_cr(j).oop)) = 1; % predict as positive to the right of OOP, negative to the left of OOP        
    cm = computeCM(y(idx_train), y_pred); % compute confusion matrix 
    for k=1:n_metrics
        train_perf(j,k) = evaluate_metrics([],[],cm.tp,cm.tn,cm.fp,cm.fn, parameter.metrics(k), parameter.metrics_predisposition(k)); % get training test set performance evaluations, considering equally positive and negative class.
    end

    y_pred = zeros(sum(idx_vali), 1, 'logical'); 
    y_pred((Cdb.Q(idx_vali) >= model_cr(j).oop)) = 1; % predict as positive to the right of OOP, negative to the left of OOP        
    cm = computeCM(y(idx_vali), y_pred); % compute confusion matrix 
    for k=1:n_metrics
        vali_perf(j,k) = evaluate_metrics([],[],cm.tp,cm.tn,cm.fp,cm.fn, parameter.metrics(k), parameter.metrics_predisposition(k)); % get training test set performance evaluations, considering equally positive and negative class.
    end
end
Cdb.train_perf = train_perf;
Cdb.vali_perf = vali_perf;
Cdb.mean_train_perf = mean(train_perf,1,'omitnan'); % average performance evaluations on train set across all cross-validations
Cdb.mean_vali_perf = mean(vali_perf,1,'omitnan'); % average performance evaluations on training test set across all cross-validations
Cdb.oop = mean([model_cr.oop],"all"); % average OOP across all cross-validations
Cdb.ps = mean([model_cr.ps],"all"); % average performance score across all cross-validations
end

function trained = train_oop(Q, y, parameter)
N_q = length(y);
[~, idx_sorted] = sort(Q);
y_sorted = y(idx_sorted);

cm0 = computeCM(y_sorted, ones(size(y_sorted), 'logical'));
comp_tmpl = y_sorted == 0;
N1 = comp_tmpl(1:end-1)';
N2 = ~ N1;
cm_array = [cm0.tn, cm0.fp, cm0.tp, cm0.fn]' + cumsum([N1; -N1; -N2; N2], 2);

cut_performance = nan(N_q-1, length(parameter.ps_metrics));
for k = 1:length(parameter.ps_metrics)
    cut_performance(:,k) = evaluate_metrics([],[],cm_array(3,:),cm_array(1,:),cm_array(2,:),cm_array(4,:), parameter.ps_metrics(k), parameter.ps_metrics_predisposition(k));
end
cut_performance_score = mean(cut_performance, 2);
best_cut_performance_score = max(cut_performance_score, [], "omitnan"); % compute performance score from performance evaluations

idx_max_cut = find(cut_performance_score == best_cut_performance_score);
N_max_cut = numel(idx_max_cut);
idx_max = idx_max_cut * 1;
if N_max_cut == 1
    oop = (Q(idx_sorted(idx_max), 1) + Q(idx_sorted(idx_max + 1), 1)) / 2;
elseif N_max_cut > 1 && rem(N_max_cut , 2) == 1
    oop = (Q(idx_sorted(median(idx_max)), 1) + Q(idx_sorted(median(idx_max) +1 ), 1)) / 2;
elseif N_max_cut > 1 && rem(N_max_cut, 2) == 0
    oop_left = (Q(idx_sorted(idx_max(N_max_cut / 2)), 1) + Q(idx_sorted(idx_max(N_max_cut / 2) + 1), 1)) / 2;
    oop_right = (Q(idx_sorted(idx_max(N_max_cut / 2 + 1)), 1) + Q(idx_sorted(idx_max(N_max_cut / 2 + 1) + 1), 1)) / 2;
    oop = (oop_left + oop_right) / 2;
end

trained.ps = best_cut_performance_score;
trained.oop = oop;
end


function cm = computeCM(y,y_pred)
cm = allocateCM();
cm.tp = sum(y_pred == 1 & y == 1);
cm.tn = sum(y_pred == 0 & y == 0);
cm.fp = sum(y_pred == 1 & y == 0);
cm.fn = sum(y_pred == 0 & y == 1);
end


function Cda = finalizeCda(X, y, Cda, metadata, parameter)
ps_list = nan(1, parameter.N_null_model + 1);
ps_list(end) = Cda.ps;
Cdb = allocateCdb(metadata, parameter);
best_random_Cdb = allocateCdb(metadata, parameter);
best_ps = Cda.ps;

rng(10086)
theta = (rand(parameter.N_null_model, 1) - 0.5) * 2 * 90; 
if strcmp(parameter.type, 'linear')
    Cda.Q = X * Cda.v';
else

end
for i = 1:parameter.N_null_model
    Cdb.Q = cosd(theta(i)) * Cda.Cdb1.Q +  sind(theta(i)) * Cda.Cdb2.Q_orth;
    Cdb = train_oop_cv(Cdb, y, parameter); 
    ps_list(i) = Cdb.ps;
    if Cdb.ps > best_ps
        best_random_Cdb = Cdb;
        best_ps = Cdb.ps;
        best_theta = theta(i);
    end
end
[~, idx_sorted] = sort(ps_list);
idx = find(idx_sorted == (parameter.N_null_model + 1));
Cda.p = 1 - (idx - 1) * ((1 / parameter.N_null_model));

if Cda.bestIter < 10,   iter = 0;   else,    iter = 10;  end
while Cda.p ~= 0 && iter < 30
iter = iter + 10;
Cda = BO_search(X, y, Cda, iter, Cda.bestIter, parameter);
ps_list(end) = Cda.ps;
[~, idx_sorted] = sort(ps_list);
idx = find(idx_sorted == (parameter.N_null_model + 1));
Cda.p = 1 - (idx - 1) * ((1 / parameter.N_null_model));
Cda.refine_BO_numpoint = iter;
end

if Cda.p ~= 0
    fields = fieldnames(Cdb);
    for i = 1:length(fields)
        Cda.(fields{i}) = best_random_Cdb.(fields{i});
    end
    Cda.theta = best_theta;
    Cda.v = cosd(best_theta) * Cda.Cdb1.v + sind(best_theta) * Cda.Cdb2.v_orth;
    if ~strcmp(parameter.type, 'linear')        
        Cda.beta = cosd(best_theta) * Cda.Cdb1.beta + sind(best_theta) * Cda.Cdb2.beta_orth;
    end
    Cda.finalization = 'best random';
else
    Cda.finalization = 'BO';
end
end


%% Memory preallocation

function cm = allocateCM()
cm.tp = zeros(1,"uint64");
cm.tn = zeros(1,"uint64");
cm.fp = zeros(1,"uint64");
cm.fn = zeros(1,"uint64");
end


function Cdb = allocateCdb(metadata, parameter)     % basic centroid-vector struct
Cdb.v = nan(1, metadata.M);
Cdb.oop = NaN;
Cdb.Q = nan(metadata.N, 1);
Cdb.beta = nan(metadata.N, 1);
Cdb.train_perf = nan(parameter.g, length(parameter.metrics));
Cdb.vali_perf = nan(parameter.g, length(parameter.metrics));
Cdb.mean_train_perf = nan(1, length(parameter.metrics));
Cdb.mean_vali_perf = nan(1, length(parameter.metrics));
end

function Cdb2 = allocateCdb2(metadata, parameter)
Cdb2 = allocateCdb(metadata, parameter);
Cdb2.v_orth = nan(1, metadata.M);
Cdb2.beta_orth = nan(metadata.N, 1);
Cdb2.Q_orth = nan(metadata.N, 1);
end

function Cda = allocateCda(metadata, parameter)
Cda = allocateCdb(metadata, parameter);
Cda.Cdb1 = allocateCdb(metadata, parameter);
Cda.Cdb2 = allocateCdb2(metadata, parameter);
Cda.theta = NaN;
Cda.test_perf = nan(1, length(parameter.metrics));
Cda.bestIter = zeros(1,'uint8');
Cda.totalIter = zeros(1,'uint8');
Cda.p = NaN;
Cda.refine_BO_numpoint = NaN;
Cda.finalization = repmat(' ', 1, 11);
Cda.parameter = parameter;
Cda.PosNegClass = cell(2,1);
end


%% Set CDA Default Parameters
function parameter = setDefaultCdaParameter(parameter)
if ~isfield(parameter, "maxIter"),                      parameter.maxIter=50;           end
if ~isfield(parameter, "cv_threshold"),                 parameter.cv_threshold=0.001;   end
if ~isfield(parameter, "N_null_model"),                 parameter.N_null_model=100;     end
if ~isfield(parameter, "ps_trace_length"),              parameter.ps_trace_length=10;   end
if ~isfield(parameter, "metrics"),                      parameter.metrics=["AUROC","AUPR","Fscore","ACscore","acc"]; end
if ~isfield(parameter, "metrics_predisposition"),       parameter.metrics_predisposition=["average","average","average","average","average"]; end
if ~isfield(parameter, "ps_metrics"),                   parameter.ps_metrics = ["Fscore","Fscore","ACscore"]; end
if ~isfield(parameter, "ps_metrics_predisposition"),    parameter.ps_metrics_predisposition = ["positive","negative","positive"]; end
if ~isfield(parameter, "g"),                            parameter.g = 5;                end
if ~isfield(parameter, "parallel"),                     parameter.parallel=0;           end
if ~isfield(parameter, "seeds"),                        parameter.seeds=cell2mat(arrayfun(@(x)x.seeds,load('data/cr_seeds.mat','seeds'),'UniformOutput',false)); end
end
