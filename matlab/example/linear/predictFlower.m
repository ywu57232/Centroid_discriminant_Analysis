
load flower_data.mat
load flower_trained.mat

classifier.cda.src = 'cda';
classifier.cda.parameter.type = 'linear';

parameter=[];
parameter.train_mode = 'multiclass-predict';
metadata = [];

rng(10086)
idx = randi(length(y), 10, 1);
XtoPred = X(idx, :);
[trained, multipredicted, parameter] = train_dataset(X, y, XtoPred, [], trained, classifier, metadata, parameter);

disp("Predicted Labels: ")
disp(multipredicted.cda.y_pred')