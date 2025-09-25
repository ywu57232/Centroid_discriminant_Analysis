dbstop if error
load MNIST_data.mat

classifier.cda.src = 'cda';
classifier.cda.parameter.type = 'linear';
trained.cda = [];

% classifier.svm.src = 'svm';
% trained.svm = [];

parameter=[];
parameter.train_mode = 'train_and_multipredict';
metadata = [];
[trained, multipredicted, parameter] = train_dataset(X, y, [], [], trained, classifier, metadata, parameter);

clf = fieldnames(classifier);
for k = 1:length(clf)
disp(['Classifier: ',clf{k}])

disp(newline)
disp("Average Binary Classification Performance: ")
mean_perf = mean(vertcat(trained.(clf{k}).model.test_perf), 1);
ste_perf = std(vertcat(trained.(clf{k}).model.test_perf), 1) / sqrt(length(trained.(clf{k}).model));
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