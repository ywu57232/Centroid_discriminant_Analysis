
load flower_data.mat

classifier.cda.src = 'cda';
classifier.cda.parameter.type = 'linear'; % kernel
classifier.cda.parameter.kernel_c = 1;
trained.cda = [];

parameter=[];
parameter.train_mode = 'train_and_multipredict';
metadata = [];

[trained, multipredicted, parameter] = train_dataset(X, y, [], [], trained, classifier, metadata, parameter);


disp(newline)
disp("Average Binary Classification Performance: ")
mean_perf = mean(vertcat(trained.cda.model.test_perf), 1);
for i = 1:length(trained.cda.parameter.metrics)
    disp([char(trained.cda.parameter.metrics(i)), ': ',num2str(mean_perf(i))])
end

if strcmp(parameter.train_mode, 'train_and_multipredict')
    disp(newline)
    disp("Average Multiclass Prediction Performance: ")
    mean_perf = mean(vertcat(multipredicted.cda.test_perf), 1);
    for i = 1:length(trained.cda.parameter.metrics)
        disp([char(trained.cda.parameter.metrics(i)), ': ',num2str(multipredicted.cda.mean_test_perf(i))])
    end
end