import numpy as np
from collections import defaultdict
from cda import cda
from evaluate_metrics import evaluate_metrics


def train_dataset(X, y, XtoPred, trained, classifier, parameter):
    y = [str(lbl) for lbl in y]

    parameter = setDefaultParameter(parameter)

    metadata = get_metadata(X, y, {}, parameter)

    multipredicted = {}

    clf_names = list(classifier.keys())
    for clf_key in clf_names:
        if "parameter" not in classifier[clf_key]:
            classifier[clf_key]["parameter"] = {}
        classifier[clf_key]["name"] = clf_key

        if not trained:
            trained[clf_key] = {}
            trained[clf_key]["model"] = train_and_test(
                X, y, metadata, classifier[clf_key], parameter
            )

            trained[clf_key]["parameter"] = trained[clf_key]["model"][0]["parameter"]
            for m in trained[clf_key]["model"]:
                m.pop("parameter", None)

        if parameter["train_mode"] == "train_and_multipredict":
            test_indices = []
            for cls_idx in range(metadata["numClass"]):
                class_loc = metadata["classLoc"][cls_idx]
                fold_indices = metadata["indices"][cls_idx]
                test_indices_part = [class_loc[i] for i in range(len(class_loc))
                                     if fold_indices[i] == 1]
                test_indices.extend(test_indices_part)
            test_indices = np.array(test_indices, dtype=int)

            X_test = X[test_indices, :]
            y_test = [y[idx] for idx in test_indices]

            multipredicted[clf_key] = multipredict(
                X_test, y_test, metadata, trained[clf_key], clf_key, parameter
            )
        elif parameter["train_mode"] == "multiclass-predict":
            multipredicted[clf_key] = multipredict(
                XtoPred, [], metadata, trained[clf_key], clf_key, parameter
            )

    trained["parameter"] = parameter
    return trained, multipredicted, parameter


def train_and_test(X, y, metadata, classifier, parameter):
    print("Start training ...")

    models = []
    k = 1
    num_pairs = metadata["numClass"] * (metadata["numClass"] - 1) // 2

    for i in range(metadata["numClass"] - 1):
        for j in range(i + 1, metadata["numClass"]):
            train_data, test_data = create_binary_pair(X, y, metadata, i, j, parameter)

            model = cda(
                train_data["X"], train_data["y"],
                positiveClass=metadata["uy"][i],
                parameter=classifier["parameter"]
            )
            if parameter["data_mode"] != "train_all_data":
                model = predict_from_model(test_data["X"], test_data["y"], classifier, model, model["parameter"])

            models.append(model)
            print(f"Trained {k} of {num_pairs} binary-pairs.")
            k += 1

    print("Finish training.")
    return models


def predict_from_model(X, y, classifier, trained_model, parameter):
    y_str = [str(lbl) for lbl in y]
    pos_class, neg_class = trained_model["PosNegClass"]
    y_tok = np.array([1 if label == pos_class else 0 for label in y_str], dtype=int)

    if classifier["name"] == "cda":
        v = trained_model["v"]
        oop = trained_model["oop"]
        Q = X @ v.T
        y_pred_tok = np.zeros(len(y_tok), dtype=int)
        y_pred_tok[Q >= oop] = 1

    cm = computeCM(y_tok, y_pred_tok)

    trained_model["test_perf"] = []
    metrics_list = parameter.get("metrics", [])
    metrics_predisp = parameter.get("metrics_predisposition", [])

    for i, metric in enumerate(metrics_list):
        predisposition = metrics_predisp[i] if i < len(metrics_predisp) else None
        score = evaluate_metrics(None, None, cm, metric, predisposition)
        trained_model["test_perf"].append(score)

    return trained_model


def create_binary_pair(X, y, metadata, i, j, parameter):
    if parameter["data_mode"] == "train-test":
        test_indices = []
        test_indices.extend([metadata["classLoc"][i][idx]
                             for idx in range(len(metadata["classLoc"][i]))
                             if metadata["indices"][i][idx] == 1])
        test_indices.extend([metadata["classLoc"][j][idx]
                             for idx in range(len(metadata["classLoc"][j]))
                             if metadata["indices"][j][idx] == 1])
    else:
        test_indices = []

    all_class_i = set(metadata["classLoc"][i])
    all_class_j = set(metadata["classLoc"][j])
    union_ij = all_class_i.union(all_class_j)
    train_indices = list(union_ij - set(test_indices))

    train_data = {
        "X": X[train_indices, :],
        "y": [y[idx] for idx in train_indices]
    }
    test_data = {
        "X": X[test_indices, :],
        "y": [y[idx] for idx in test_indices]
    }
    return train_data, test_data


def multipredict(X, y, metadata, classifier, clf_name, parameter):
    num_samples = X.shape[0]
    num_models = len(classifier["model"])
    Q = np.full((num_samples, num_models), np.nan)

    for k in range(num_models):
        model_k = classifier["model"][k]
        if clf_name == "cda":
            v = model_k["v"]
            oop = model_k["oop"]
            Q[:, k] = (X @ v.T) - oop
        else:
            Q[:, k] = 0.0

    coding_matrix = np.zeros((metadata["numClass"], num_models), dtype=float)

    k = 0
    for i in range(metadata["numClass"] - 1):
        for j in range(i + 1, metadata["numClass"]):
            pos_neg = classifier["model"][k]["PosNegClass"]
            class_i_str = metadata["uy"][i]
            class_j_str = metadata["uy"][j]
            pos_str = pos_neg[0]
            coding_matrix[i, k] = 1.0 if (class_i_str == pos_str) else -1.0
            coding_matrix[j, k] = 1.0 if (class_j_str == pos_str) else -1.0
            k += 1

    total_hinge_loss = np.full((num_samples, metadata["numClass"]), np.nan)
    for C in range(metadata["numClass"]):
        hinge_vals = np.maximum(0, -coding_matrix[C, :] * Q)
        total_hinge_loss[:, C] = 0.5 * np.sum(hinge_vals, axis=1)

    idx_min = np.argmin(total_hinge_loss, axis=1)
    y_pred = [metadata["uy"][ix] for ix in idx_min]

    if parameter["train_mode"] == "multiclass-predict":
        return {"y_pred": y_pred}

    y_true = y
    unique_classes = metadata["uy"]

    class2idx = {cls_label: i for i, cls_label in enumerate(unique_classes)}

    y_true_idx = np.array([class2idx[label] for label in y_true], dtype=int)
    y_pred_idx = np.array([class2idx[label] for label in y_pred], dtype=int)

    CM = np.zeros((metadata["numClass"], metadata["numClass"]), dtype=int)
    for i_sample in range(len(y_true_idx)):
        CM[y_true_idx[i_sample], y_pred_idx[i_sample]] += 1

    multipredicted = defaultdict(list)
    cm_list = []
    for C in range(metadata["numClass"]):
        tp = CM[C, C]
        tn = np.sum(CM[np.arange(metadata["numClass"]) != C][:,
                    np.arange(metadata["numClass"]) != C])
        fp = np.sum(CM[np.arange(metadata["numClass"]) != C, C])
        fn = np.sum(CM[C, np.arange(metadata["numClass"]) != C])

        class_cm = {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        }
        cm_list.append(class_cm)

        perfs = []
        metrics_list = classifier["parameter"].get("metrics", [])
        metrics_predisp = classifier["parameter"].get("metrics_predisposition", [])
        for i_metric, metric in enumerate(metrics_list):
            predisposition = metrics_predisp[i_metric] if i_metric < len(metrics_predisp) else None
            score = evaluate_metrics(None, None, class_cm, metric, predisposition)
            perfs.append(score)
        multipredicted.setdefault("test_perf", []).append(perfs)

    multipredicted["test_perf"] = np.array(multipredicted["test_perf"], dtype=float)
    multipredicted["mean_test_perf"] = np.nanmean(multipredicted["test_perf"], axis=0)

    return dict(multipredicted)


def get_metadata(X, y, metadata, parameter):
    uy = []
    class_map = {}
    for label in y:
        if label not in class_map:
            class_map[label] = len(uy)
            uy.append(label)
    num_class = len(uy)

    N, M = X.shape
    num_per_class = [0] * num_class
    for label in y:
        num_per_class[class_map[label]] += 1

    classLoc = [[] for _ in range(num_class)]
    for i, label in enumerate(y):
        classLoc[class_map[label]].append(i)

    # Now do random folds
    indices = []
    rng_seed = parameter["assignSeed"]
    for c_idx in range(num_class):
        n_samples_c = len(classLoc[c_idx])
        np.random.seed(rng_seed)
        rng_seed += 1
        fold_assignment = np.random.randint(1, 6, size=n_samples_c)
        indices.append(fold_assignment)

    return {
        "uy": uy,
        "numClass": num_class,
        "N": N,
        "M": M,
        "numPerClass": num_per_class,
        "classLoc": classLoc,
        "indices": indices
    }


def computeCM(y_true_tok, y_pred_tok):
    cm = allocateCM()
    cm["tp"] = int(np.sum((y_pred_tok == 1) & (y_true_tok == 1)))
    cm["tn"] = int(np.sum((y_pred_tok == 0) & (y_true_tok == 0)))
    cm["fp"] = int(np.sum((y_pred_tok == 1) & (y_true_tok == 0)))
    cm["fn"] = int(np.sum((y_pred_tok == 0) & (y_true_tok == 1)))
    return cm


def allocateCM():
    return {"tp": 0, "tn": 0, "fp": 0, "fn": 0}


def setDefaultParameter(parameter):
    if "train_mode" not in parameter:
        parameter["train_mode"] = "train_pairs"
    if "data_mode" not in parameter:
        parameter["data_mode"] = "train-test"
    if "trained_model" not in parameter:
        parameter["trained_model"] = None
    if "assignSeed" not in parameter:
        parameter["assignSeed"] = 10086
    return parameter

