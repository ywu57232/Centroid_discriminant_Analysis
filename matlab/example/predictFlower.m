
load flower_data.mat
load flower_trained.mat

classifier.cda.src = 'cda';
parameter=[];
parameter.train_mode = 'multiclass-predict';

rng(10086)
idx = randi(length(y), 10, 1);
XtoPred = X(idx, :);
[trained, multipredicted, parameter] = train_dataset(X, y, XtoPred, trained, classifier, parameter);

disp("Predicted Labels: ")
disp(multipredicted.cda.y_pred')