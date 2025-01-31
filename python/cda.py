import numpy as np
import time
from evaluate_metrics import evaluate_metrics
from BOImpl import BOImpl

def cda(X, y, positiveClass, parameter):
    start_time = time.time()

    y = [str(label) for label in y]

    uy = sorted(list(set(y)))
    if len(uy) != 2:
        raise ValueError("Input data must have exactly 2 distinct classes.")

    if positiveClass is None or len(str(positiveClass)) == 0:
        positiveClass = y[0]
    else:
        positiveClass = str(positiveClass)

    if positiveClass not in uy:
        raise ValueError(f"Specified positiveClass '{positiveClass}' not found in y.")
    neg_classes = [cls for cls in uy if cls != positiveClass]
    PosNegClass = [positiveClass] + neg_classes  # e.g. [pos, neg]

    y_bool = np.array([1 if label == positiveClass else 0 for label in y], dtype=bool)

    parameter = setDefaultCdaParameter(parameter)

    N, M = X.shape
    metadata = {
        "N": N,
        "M": M
    }

    alpha = np.ones(N) / np.sqrt(N)

    Cda = allocateCda(metadata, parameter)

    Cda["Cdb1"]["v"] = soft_centroid_vector(X, y_bool, alpha)
    Cda["Cdb1"]["Q"] = X @ Cda["Cdb1"]["v"].T
    Cda["Cdb1"] = train_oop_cv(Cda["Cdb1"], y_bool, parameter)

    iter_count = 1
    max_ps = 0.0
    bestCda = {}
    ps_trace = [np.nan] * parameter["ps_trace_length"]

    while iter_count <= parameter["maxIter"]:
        alpha = update_sample_weight(alpha, X, y_bool, Cda["Cdb1"], parameter)

        Cda["Cdb2"]["v"] = soft_centroid_vector(X, y_bool, alpha)
        v_orth = Cda["Cdb2"]["v"] - (Cda["Cdb1"]["v"] *
                                     (Cda["Cdb2"]["v"] @ Cda["Cdb1"]["v"]))
        v_orth_norm = np.linalg.norm(v_orth)
        if v_orth_norm == 0:
            Cda["Cdb2"]["v_orth"] = np.zeros_like(Cda["Cdb2"]["v"])
        else:
            Cda["Cdb2"]["v_orth"] = v_orth / v_orth_norm

        Cda["Cdb2"]["Q_orth"] = X @ Cda["Cdb2"]["v_orth"].T

        Cda = BO_search(X, y_bool, Cda, min(10, iter_count + 3), iter_count, parameter)

        ps_trace[:-1] = ps_trace[1:]
        ps_trace[-1] = Cda["ps"]
        if Cda["ps"] > max_ps:
            max_ps = Cda["ps"]
            Cda["bestIter"] = iter_count
            bestCda = dict(Cda)

        bestCda["totalIter"] = iter_count

        if not any(np.isnan(ps_trace)):
            arr = np.array(ps_trace)
            cv_val = np.std(arr) / np.mean(arr)
            if cv_val < parameter["cv_threshold"]:
                break

        for field_name in Cda["Cdb1"].keys():
            Cda["Cdb1"][field_name] = Cda[field_name]

        iter_count += 1

    Cda = bestCda

    Cda = finalizeCda(X, y_bool, Cda, metadata, parameter)

    Cda["PosNegClass"] = PosNegClass
    end_time = time.time()
    Cda["runtime"] = end_time - start_time

    return Cda


def BO_search(X, y, Cda, BOIter, CDAIter, parameter):
    def fun(var_theta):
        return lossfun(var_theta, y, Cda, parameter)


    var_theta_opt = BOImpl(fun, [-90, 90], BOIter, CDAIter)

    Cda["theta"] = var_theta_opt["XAtMinEstimatedObjective"]
    angle_rad = np.deg2rad(Cda["theta"])
    Cda["v"] = (Cda["Cdb1"]["v"] * np.cos(angle_rad)
               + Cda["Cdb2"]["v_orth"] * np.sin(angle_rad))
    Cda["Q"] = X @ Cda["v"].T
    Cda = train_oop_cv(Cda, y, parameter)
    return Cda


def lossfun(var_theta, y, Cda, parameter):
    angle_rad = np.deg2rad(var_theta)
    Q_candidate = (Cda["Cdb1"]["Q"] * np.cos(angle_rad)
                   + Cda["Cdb2"]["Q_orth"] * np.sin(angle_rad))
    tempCdb = dict(Cda)
    tempCdb["Q"] = Q_candidate
    tempCdb = train_oop_cv(tempCdb, y, parameter)
    return -tempCdb["ps"]


def soft_centroid_vector(X, y_bool, alpha):
    neg_sum = np.sum(alpha[~y_bool])
    pos_sum = np.sum(alpha[y_bool])
    sign_map = np.where(y_bool, 1.0, -1.0)
    weights = []
    for i, sgn in enumerate(sign_map):
        if sgn > 0:
            w = alpha[i] / pos_sum
        else:
            w = -alpha[i] / neg_sum
        weights.append(w)
    weights_arr = np.array(weights)

    v = weights_arr.T @ X
    if not np.any(v):
        raise ValueError("Cannot trace the vector from overlapped group centroids.")
    # Normalize
    norm_v = np.linalg.norm(v)
    v = v / norm_v
    return v


def update_sample_weight(alpha, X, y_bool, Cdb, parameter):
    Q = X @ Cdb["v"].T
    temp = train_oop_cv(Cdb, y_bool, parameter)
    oop_val = temp["oop"]
    d = np.abs(Q - oop_val)
    d1 = d / np.sum(d)

    d2 = np.abs(d1 - np.min(d1) - np.max(d1))
    alpha_new = alpha * d2
    alpha_norm = alpha_new / np.linalg.norm(alpha_new)
    return alpha_norm


def train_oop_cv(Cdb, y, parameter):
    Q = Cdb["Q"]
    N = len(y)

    n_metrics = len(parameter["metrics"])
    train_perf = np.full((parameter["g"], n_metrics), np.nan)
    vali_perf = np.full((parameter["g"], n_metrics), np.nan)

    model_cr = []
    np.random.seed(parameter["seeds"][0])

    indices = crossvalind_kfold(N, parameter["g"])
    for j in range(parameter["g"]):
        idx_vali = (indices == j)
        idx_train = ~idx_vali

        partialCdb = {"Q": Q[idx_train]}
        trained_oop = train_oop(partialCdb["Q"], y[idx_train], N, parameter)
        model_cr.append(trained_oop)

        Q_train = Q[idx_train]
        y_pred = (Q_train >= trained_oop["oop"]).astype(bool)
        cm_train = computeCM(y[idx_train], y_pred)
        for k_m in range(n_metrics):
            metric_name = parameter["metrics"][k_m]
            predisposition = parameter["metrics_predisposition"][k_m]
            train_perf[j, k_m] = evaluate_metrics(None, None, cm_train, metric_name, predisposition)

        # Evaluate performance on validation set
        Q_vali = Q[idx_vali]
        y_pred_vali = (Q_vali >= trained_oop["oop"]).astype(bool)
        cm_vali = computeCM(y[idx_vali], y_pred_vali)
        for k_m in range(n_metrics):
            metric_name = parameter["metrics"][k_m]
            predisposition = parameter["metrics_predisposition"][k_m]
            vali_perf[j, k_m] = evaluate_metrics(None, None, cm_vali, metric_name, predisposition)

    # gather means
    Cdb["train_perf"] = train_perf
    Cdb["vali_perf"] = vali_perf
    Cdb["mean_train_perf"] = np.nanmean(train_perf, axis=0)
    Cdb["mean_vali_perf"] = np.nanmean(vali_perf, axis=0)

    # average OOP and performance score across all folds
    oop_list = [m["oop"] for m in model_cr]
    ps_list = [m["ps"] for m in model_cr]
    Cdb["oop"] = float(np.mean(oop_list))
    Cdb["ps"] = float(np.mean(ps_list))
    return Cdb


def train_oop(Q, y, N, parameter):

    N_q = len(y)
    N_cut = int(np.floor(np.sqrt(N)))
    Int = N_q // N_cut

    idx_sorted = np.argsort(Q)
    y_sorted = y[idx_sorted]


    cm_list = []
    y_pred = np.ones(N_q, dtype=bool)
    y_pred[: Int * 1] = False
    cm_list.append(computeCM(y_sorted, y_pred))
    for i in range(2, N_cut):

        cm_next = updateCM_forward(dict(cm_list[-1]), y_sorted, i, Int)
        cm_list.append(cm_next)

    cut_perf = []
    for cm_idx, cm_dict in enumerate(cm_list):
        perf_vals = []
        for metric_name, predisposition in zip(parameter["ps_metrics"], parameter["ps_metrics_predisposition"]):
            val = evaluate_metrics(None, None, cm_dict, metric_name, predisposition)
            perf_vals.append(val)
        cut_perf.append(perf_vals)
    cut_perf_arr = np.array(cut_perf, dtype=float)
    cut_perf_score = np.mean(cut_perf_arr, axis=1)

    best_cut_performance_score = np.max(cut_perf_score)
    idx_max_cut = np.where(cut_perf_score == best_cut_performance_score)[0]

    idx_max = (idx_max_cut + 1) * Int

    if len(idx_max) == 1:
        i_cut = idx_max[0]
        if i_cut + 1 < len(idx_sorted):
            oop_val = 0.5 * (Q[idx_sorted[i_cut]] + Q[idx_sorted[i_cut + 1]])
        else:
            oop_val = Q[idx_sorted[i_cut]]
    else:
        if len(idx_max) % 2 == 1:
            mid_index = idx_max[len(idx_max)//2]
            if mid_index + 1 < N_q:
                oop_val = 0.5 * (Q[idx_sorted[mid_index]] + Q[idx_sorted[mid_index + 1]])
            else:
                oop_val = Q[idx_sorted[mid_index]]
        else:
            half = len(idx_max)//2
            left_index = idx_max[half - 1]
            right_index = idx_max[half]
            if right_index + 1 < N_q:
                oop_left = 0.5 * (Q[idx_sorted[left_index]] + Q[idx_sorted[left_index + 1]])
                oop_right = 0.5 * (Q[idx_sorted[right_index]] + Q[idx_sorted[right_index + 1]])
                oop_val = 0.5 * (oop_left + oop_right)
            else:
                oop_val = Q[idx_sorted[idx_max[0]]]

    trained = {
        "ps": best_cut_performance_score,
        "oop": oop_val
    }
    return trained


def updateCM_forward(r, mapped_labels, i, Int):
    cm = dict(r)

    start_idx = (i - 1) * Int
    end_idx = i * Int

    for j in range(start_idx, min(end_idx, len(mapped_labels))):
        if mapped_labels[j] == 0:
            cm["tn"] += 1
            cm["fp"] -= 1
        else:
            cm["tp"] -= 1
            cm["fn"] += 1

    return cm


def computeCM(y_true, y_pred):
    cm = allocateCM()
    y_true = np.array(y_true, dtype=bool)
    y_pred = np.array(y_pred, dtype=bool)

    cm["tp"] = int(np.sum((y_true == 1) & (y_pred == 1)))
    cm["tn"] = int(np.sum((y_true == 0) & (y_pred == 0)))
    cm["fp"] = int(np.sum((y_true == 0) & (y_pred == 1)))
    cm["fn"] = int(np.sum((y_true == 1) & (y_pred == 0)))
    return cm


def finalizeCda(X, y, Cda, metadata, parameter):
    ps_list = np.full(parameter["N_null_model"] + 1, np.nan)
    ps_list[-1] = Cda["ps"]

    best_ps = Cda["ps"]
    best_random_Cdb = None
    best_theta = 0.0

    # Build placeholders
    Q_Cdb1 = X @ Cda["Cdb1"]["v"].T
    Q_Cdb2_orth = X @ Cda["Cdb2"]["v_orth"].T

    # random angles
    np.random.seed(10086)
    theta_vals = (np.random.rand(parameter["N_null_model"]) - 0.5) * 2 * 90

    # baseline struct for random tries
    Cdb = allocateCdb(metadata, parameter)

    for i, th in enumerate(theta_vals):
        angle_rad = np.deg2rad(th)
        Q_rand = (np.cos(angle_rad) * Q_Cdb1) + (np.sin(angle_rad) * Q_Cdb2_orth)
        Cdb["Q"] = Q_rand
        Cdb = train_oop_cv(Cdb, y, parameter)
        ps_list[i] = Cdb["ps"]
        if Cdb["ps"] > best_ps:
            best_random_Cdb = dict(Cdb)
            best_ps = Cdb["ps"]
            best_theta = th

    # compute p-value
    idx_sorted = np.argsort(ps_list)
    index_of_model = np.where(idx_sorted == (parameter["N_null_model"]))[0][0]
    Cda["p"] = 1.0 - (index_of_model) * (1.0 / parameter["N_null_model"])


    n_iter = 0
    while Cda["p"] != 0 and n_iter < 30:
        n_iter += 10
        # More BO search
        Cda = BO_search(X, y, Cda, n_iter, Cda["bestIter"], parameter)
        ps_list[-1] = Cda["ps"]
        idx_sorted = np.argsort(ps_list)
        index_of_model = np.where(idx_sorted == (parameter["N_null_model"]))[0][0]
        Cda["p"] = 1.0 - (index_of_model) * (1.0 / parameter["N_null_model"])
        Cda["refine_BO_numpoint"] = n_iter

    if Cda["p"] != 0 and best_random_Cdb is not None:
        # If random is better, we adopt that
        for field_name in best_random_Cdb.keys():
            Cda[field_name] = best_random_Cdb[field_name]
        Cda["theta"] = best_theta
        angle_rad = np.deg2rad(best_theta)
        Cda["v"] = (np.cos(angle_rad) * Cda["Cdb1"]["v"]
                   + np.sin(angle_rad) * Cda["Cdb2"]["v_orth"])
        Cda["finalization"] = "best random"
    else:
        Cda["finalization"] = "BO"

    return Cda


def allocateCM():
    return {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0
    }

def allocateCdb(metadata, parameter):
    M = metadata["M"]
    N = metadata["N"]
    n_metrics = len(parameter["metrics"])
    return {
        "v": np.full((M,), np.nan),
        "oop": np.nan,
        "Q": np.full((N,), np.nan),
        "train_perf": np.full((parameter["g"], n_metrics), np.nan),
        "vali_perf": np.full((parameter["g"], n_metrics), np.nan),
        "mean_train_perf": np.full((n_metrics,), np.nan),
        "mean_vali_perf": np.full((n_metrics,), np.nan)
    }

def allocateCdb2(metadata, parameter):
    cdb = allocateCdb(metadata, parameter)
    cdb["v_orth"] = np.full((metadata["M"],), np.nan)
    cdb["Q_orth"] = np.full((metadata["N"],), np.nan)
    return cdb

def allocateCda(metadata, parameter):
    cda_struct = allocateCdb(metadata, parameter)
    cda_struct["Cdb1"] = allocateCdb(metadata, parameter)
    cda_struct["Cdb2"] = allocateCdb2(metadata, parameter)
    cda_struct["theta"] = np.nan
    cda_struct["test_perf"] = np.full((1, len(parameter["metrics"])), np.nan)
    cda_struct["bestIter"] = 0
    cda_struct["totalIter"] = 0
    cda_struct["p"] = np.nan
    cda_struct["refine_BO_numpoint"] = np.nan
    cda_struct["finalization"] = "          "
    cda_struct["parameter"] = parameter
    cda_struct["PosNegClass"] = [None, None]
    return cda_struct


def setDefaultCdaParameter(parameter):
    if "maxIter" not in parameter:
        parameter["maxIter"] = 50
    if "cv_threshold" not in parameter:
        parameter["cv_threshold"] = 0.001
    if "N_null_model" not in parameter:
        parameter["N_null_model"] = 100
    if "ps_trace_length" not in parameter:
        parameter["ps_trace_length"] = 10
    if "metrics" not in parameter:
        parameter["metrics"] = ["AUC", "AUPR", "Fscore", "ACscore", "acc"]
    if "metrics_predisposition" not in parameter:
        parameter["metrics_predisposition"] = ["average"] * 5
    if "ps_metrics" not in parameter:
        parameter["ps_metrics"] = ["Fscore", "Fscore", "ACscore"]
    if "ps_metrics_predisposition" not in parameter:
        parameter["ps_metrics_predisposition"] = ["positive", "negative", "positive"]
    if "g" not in parameter:
        parameter["g"] = 5
    if "parallel" not in parameter:
        parameter["parallel"] = 0
    if "seeds" not in parameter:
        parameter["seeds"] = [12345, 23456, 34567, 45678, 56789]
    return parameter


def crossvalind_kfold(n_samples, k):
    arr = np.arange(n_samples)
    np.random.shuffle(arr)
    folds = np.zeros(n_samples, dtype=int)
    chunk_size = int(np.ceil(n_samples / float(k)))
    for i in range(k):
        start = i * chunk_size
        end = min(start + chunk_size, n_samples)
        folds[arr[start:end]] = i
    return folds

