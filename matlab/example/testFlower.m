
load flower_data.mat

classifier.cda.src = 'cda';
parameter=[];
parameter.train_mode = 'train_and_multipredict';

[trained, multipredicted, parameter] = train_dataset(X, y, [], [], classifier, parameter);


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
        disp([char(trained.cda.parameter.metrics(i)), ': ',num2str(mean_perf(i))])
    end
end