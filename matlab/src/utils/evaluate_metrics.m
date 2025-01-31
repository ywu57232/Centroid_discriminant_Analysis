function perf = evaluate_metrics(y, y_pred, tp, tn, fp, fn, metric, predisposition)

% Usage: 
% perf = evaluate_metrics(y, y_pred, [], metric, -) : evaluate from true
% labels y and predicted labels y_pred
% perf = evaluate_metrics([], [], cm, metric, -) : evaluate from confusion
% matrix cm

% if ~isempty(y) && ~isempty(y_pred) && isempty(cm)
%     cm = computeCM(y,y_pred);
% elseif isempty(y) && isempty(y_pred) && ~isempty(cm)
% 
% else
%     error("Use either labels or confusion matrix.")
% end

if nargin < 5
    predisposition = 'average';
end

if strcmp(metric, ["AUROC",'ACscore','acc'])
    predisposition = 'positive';
end

if strcmp(predisposition, 'positive')
    perf = compute_metric(tp, tn, fp, fn, metric);   % evaluation for positive class
elseif strcmp(predisposition, 'negative') % evaluation for negative class
    [tp, tn, fp, fn] = flipcm(tp, tn, fp, fn);
    perf = compute_metric(tp, tn, fp, fn, metric);
elseif strcmp(predisposition, 'average') % evaluation for both classes and take average
    a = compute_metric(tp, tn, fp, fn, metric);
    [tp, tn, fp, fn] = flipcm(tp, tn, fp, fn);
    b = compute_metric(tp, tn, fp, fn, metric);
    perf = mean([a,b], "omitnan"); 
end
end


function [tp, tn, fp, fn] = flipcm(tp, tn, fp, fn)  % flip both true labels and predicted labels.
tp0 = tp; tn0 = tn; fp0 = fp; fn0 = fn;
tp = tn0; tn = tp0; fp = fn0; fn = fp0;
end

function result = compute_metric(tp, tn, fp, fn, metric)
if strcmp(metric, 'AUC')
    if tp==0 && fn==0
        sens=1;
    else
        sens = tp/(tp+fn);
    end
    if fp==0 && tn==0
        fpr=0;
    else
        fpr = fp/(fp+tn);        
    end
    
    x = [0;sens;1];
    y = [0;fpr;1];
    
    result = trapz(y,x);
elseif strcmp(metric, 'AUPR')
    if tp==0 && fn==0
        sens=1;
    else
        sens = tp/(tp+fn);
    end
    if tp==0 && fp==0
        prec=0;
    else
        prec = tp/(tp+fp);
    end
    
    x = [0;sens;1];
    y = [1;prec;0];
    
    result = trapz(x,y);
elseif strcmp(metric, 'Fscore')
    sens = tp ./ (tp + fn);
    % sens(tp==0 & fn==0) = 1;

    prec = tp ./ (tp + fp);
    % prec(tp==0 & fp==0) = 0;

    result = (2 * prec .* sens) ./ (prec + sens);
    % result(sens==0 & prec==0) = 0;

    % result((tp==0 & fn==0) | (tp==0 & fp==0)) = 0;

    %%
    % if (tp==0 && fn==0) || (tp==0 && fp==0)
    %     result = 0;
    % else
    %     if tp==0 && fn==0
    %         sens=1;
    %     else
    %         sens = tp ./ (tp + fn);
    %     end
    %     if tp==0 && fp==0
    %         prec=0;
    %     else
    %         prec = tp ./ (tp + fp);
    %     end
    %     if sens==0 && prec==0
    %         result = 0;
    %     else
    %         result = (2 * prec .* sens) ./ (prec + sens);
    %     end
    % end
elseif strcmp(metric, 'ACscore')
    sens = tp ./ (tp + fn);
    % sens(tp==0 & fn==0) = 1;

    spc = tn ./ (fp + tn);
    % spc(fp==0 & tn==0) = 0;

    result = 2 * sens .* spc ./ (sens + spc);
    % result(sens == 0 & spc == 0) = 0;

    %%
    % if tp==0 && fn==0
    %     sens = 0;
    % else
    %     sens = tp/(tp+fn);
    % end
    % if fp==0 && tn==0
    %     spc = 0;
    % else
    %     spc = tn/(fp+tn);
    % end
    % if sens == 0 && spc == 0
    %     result = 0;
    % else
    %     result = 2*sens*spc/(sens+spc);
    % end
elseif strcmp(metric, 'acc')
    result = (tp + tn) / (tp + tn + fp + fn);
else
    error('Possible metrics: ''AUROC'', ''AUPR'', ''Fsore'', ''ACscore'', ''acc''.');
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