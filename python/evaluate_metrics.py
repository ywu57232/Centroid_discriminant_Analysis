import numpy as np

def evaluate_metrics(y, y_pred, cm, metric, predisposition=None):
    if y is not None and y_pred is not None and cm is None:
        cm = computeCM(np.array(y), np.array(y_pred))
    elif (y is None or y_pred is None) and cm is not None:
        pass
    else:
        raise ValueError("Use either labels (y, y_pred) or a confusion matrix (cm), not both.")

    if predisposition is None:
        predisposition = 'average'

    if metric in ["AUROC", "ACscore", "acc"]:
        predisposition = 'positive'

    if predisposition == 'positive':
        perf = compute_metric(cm["tp"], cm["tn"], cm["fp"], cm["fn"], metric)

    elif predisposition == 'negative':
        flipped = flipcm(cm)
        perf = compute_metric(flipped["tp"], flipped["tn"],
                              flipped["fp"], flipped["fn"], metric)

    elif predisposition == 'average':
        a = compute_metric(cm["tp"], cm["tn"], cm["fp"], cm["fn"], metric)
        flipped = flipcm(cm)
        b = compute_metric(flipped["tp"], flipped["tn"],
                           flipped["fp"], flipped["fn"], metric)
        perf = np.nanmean([a, b])

    else:
        raise ValueError("predisposition must be 'positive', 'negative', or 'average'.")

    return perf


def flipcm(cm):
    return {
        "tp": cm["tn"],
        "tn": cm["tp"],
        "fp": cm["fn"],
        "fn": cm["fp"]
    }


def compute_metric(tp, tn, fp, fn, metric):
    if metric == "AUC":
        if (tp == 0 and fn == 0):
            sens = 1.0
        else:
            sens = tp / (tp + fn)

        if (fp == 0 and tn == 0):
            fpr = 0.0
        else:
            fpr = fp / (fp + tn)

        x = np.array([0, sens, 1], dtype=float)
        y = np.array([0, fpr, 1], dtype=float)
        result = np.trapz(x, y)

    elif metric == "AUPR":
        if (tp == 0 and fn == 0):
            sens = 1.0
        else:
            sens = tp / (tp + fn)

        if (tp == 0 and fp == 0):
            prec = 0.0
        else:
            prec = tp / (tp + fp)

        x = np.array([0, sens, 1], dtype=float)
        y = np.array([1, prec, 0], dtype=float)
        result = np.trapz(y, x)

    elif metric == "Fscore":
        if (tp == 0 and fn == 0):
            sens = 1.0
        else:
            sens = tp / (tp + fn)
        if (tp == 0 and fp == 0):
            prec = 0.0
        else:
            prec = tp / (tp + fp)

        if sens == 0.0 and prec == 0.0:
            result = 0.0
        else:
            result = 2.0 * prec * sens / (prec + sens)

    elif metric == "ACscore":
        if (tp == 0 and fn == 0):
            sens = 0.0
        else:
            sens = tp / (tp + fn)

        if (fp == 0 and tn == 0):
            spec = 0.0
        else:
            spec = tn / (fp + tn)

        if sens == 0.0 and spec == 0.0:
            result = 0.0
        else:
            result = 2.0 * sens * spec / (sens + spec)

    elif metric == "acc":
        total = tp + tn + fp + fn
        result = (tp + tn) / total if total > 0 else 0.0

    else:
        raise ValueError("Possible metrics: 'AUC', 'AUPR', 'Fscore', 'ACscore', 'acc'.")

    return float(result)


def computeCM(y, y_pred):
    y = np.array(y, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    tp = np.sum((y_pred == 1) & (y == 1))
    tn = np.sum((y_pred == 0) & (y == 0))
    fp = np.sum((y_pred == 1) & (y == 0))
    fn = np.sum((y_pred == 0) & (y == 1))

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }
