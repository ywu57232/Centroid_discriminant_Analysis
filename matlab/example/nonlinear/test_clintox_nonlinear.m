dbstop if error

load('clintox_atomic.mat')      
X = double(X); y = double(y);

rng(10086)
rp1 = randperm(length(y));
test_idx = rp1(1:round(length(y)/5));
train_idx = rp1(round(length(y)/5) + 1 : length(y));
X_test = X(test_idx,:);
y_test = y(test_idx);
X = X(train_idx,:);
y = y(train_idx);

% linear CDA
classifier.cda.src = 'cda';
classifier.cda.BO_constraint = 'mix_two_lines_in_plane';
classifier.cda.parameter.type = 'linear'; % linear and nonlinear

% nonlinear CDA
classifier.cda_kernel.src = 'cda';
classifier.cda_kernel.BO_constraint = 'mix_two_lines_in_plane';
classifier.cda_kernel.kernel = 'rbf';
classifier.cda_kernel.kernel_sigma = 'sqrt(2 * mean_dist)';
classifier.cda_kernel.parameter.type = 'nonlinear'; % linear and nonlinear

% linear SVM
classifier.svm.src = 'svm';

% nonlinear SVM
classifier.svm_kernel_libsvm.src = 'svm_kernel_libsvm';
classifier.svm_kernel_libsvm.kernel = 'rbf';
classifier.svm_kernel_libsvm.kernel_sigma = 'sqrt(2 * mean_dist)';

parameter=[];
parameter.train_mode = 'train_and_multipredict';
metadata = [];
trained = [];
clf = fieldnames(classifier);
for k = 1:length(clf)
curr_classifier = [];
curr_classifier.(clf{k}) = classifier.(clf{k});
trained.(clf{k}) = [];

if ismember(clf(k), {'cda', 'svm'})
    [curr_trained, curr_multipredicted, parameter] = train_dataset(X, y, X_test, y_test, trained, curr_classifier, metadata, parameter);
else
    % BO
    rng(10086)
    fun=@(Var)subfun_loss(Var, X, y, X_test, y_test, trained, curr_classifier, metadata, parameter);
    c = optimizableVariable('c',[1e-1, 1e3],'Transform','log'); % [1e-1, 1e3]
    r=bayesopt(fun,[c],'IsObjectiveDeterministic',false,'AcquisitionFunctionName','expected-improvement-plus',"MaxObjectiveEvaluations",30,'Verbose',1); % ,'PlotFcn',[]  ,'InitialX',table(model_best.deg);  ,'InitialX',table([]) 
    Var = r.XAtMinEstimatedObjective;
    [curr_trained, curr_multipredicted, curr_classifier] = subfun(Var, X, y, X_test, y_test, trained, curr_classifier, metadata, parameter);
end
trained.(clf{k}) = curr_trained.(clf{k});
multipredicted.(clf{k}) = curr_multipredicted.(clf{k});
classifier.(clf{k}) = curr_classifier.(clf{k});

disp(['Classifier: ',clf{k}])

disp(newline)
disp("Average Binary Classification Performance: ")
mean_perf = mean(vertcat(trained.(clf{k}).model.test_perf), 1);
ste_perf = std(vertcat(trained.(clf{k}).model.test_perf), 0, 1) / sqrt(length(trained.(clf{k}).model));
for i = 1:length(trained.(clf{k}).parameter.metrics)
    disp([char(trained.(clf{k}).parameter.metrics(i)), ': ',num2str(mean_perf(i)),' + ',num2str(ste_perf(i),1)])
end

if strcmp(parameter.train_mode, 'train_and_multipredict')
    disp(newline)
    disp("Average Multiclass Prediction Performance: ")
    mean_perf = mean(vertcat(multipredicted.(clf{k}).test_perf), 1);
    ste_perf = std(multipredicted.(clf{k}).test_perf) / sqrt(size(multipredicted.(clf{k}).test_perf,1));
    for i = 1:length(trained.(clf{k}).parameter.metrics)
        disp([char(trained.(clf{k}).parameter.metrics(i)), ': ',num2str(multipredicted.(clf{k}).mean_test_perf(i)),' + ',num2str(ste_perf(i),1)])
    end
end
close all
end


function loss = subfun_loss(Var, X, y, X_test, y_test, trained, classifier, metadata, parameter)
[trained, multipredicted, classifier, parameter] = subfun(Var, X, y, X_test, y_test, trained, classifier, metadata, parameter);
clf = fieldnames(classifier);
mean_test_perf = multipredicted.(clf{:}).mean_test_perf;
loss = -(mean_test_perf(3)*2/3 + mean_test_perf(4)*1/3);
end

function [trained, multipredicted, classifier, parameter] = subfun(Var, X, y, X_test, y_test, trained, classifier, metadata, parameter)
clf = fieldnames(classifier);
classifier.(clf{:}).parameter.kernel_c = Var.c;

[trained, multipredicted, parameter] = train_dataset(X, y, X_test, y_test, trained, classifier, metadata, parameter);

clf = fieldnames(classifier);
for k = 1:length(clf)
disp(['Classifier: ',clf{k}])

disp(newline)
disp("Average Binary Classification Performance: ")
mean_perf = mean(vertcat(trained.(clf{k}).model.test_perf), 1);
ste_perf = std(vertcat(trained.(clf{k}).model.test_perf), 0, 1) / sqrt(length(trained.(clf{k}).model));
for i = 1:length(trained.(clf{k}).parameter.metrics)
    disp([char(trained.(clf{k}).parameter.metrics(i)), ': ',num2str(mean_perf(i)),' + ',num2str(ste_perf(i),1)])
end

if strcmp(parameter.train_mode, 'train_and_multipredict')
    disp(newline)
    disp("Average Multiclass Prediction Performance: ")
    mean_perf = mean(vertcat(multipredicted.(clf{k}).test_perf), 1);
    ste_perf = std(multipredicted.(clf{k}).test_perf) / sqrt(size(multipredicted.(clf{k}).test_perf,1));
    for i = 1:length(trained.(clf{k}).parameter.metrics)
        disp([char(trained.(clf{k}).parameter.metrics(i)), ': ',num2str(multipredicted.(clf{k}).mean_test_perf(i)),' + ',num2str(ste_perf(i),1)])
    end
end
end
end
