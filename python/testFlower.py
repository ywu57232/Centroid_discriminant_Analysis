import numpy as np
import scipy.io as sio
from train_dataset import train_dataset


def main():
    data = sio.loadmat("../matlab/data/flower_data.mat")
    X = data["X"]
    y = data["y"]

    classifier = {
        "cda": {
            "src": "cda"
        }
    }
    parameter = {}
    parameter["train_mode"] = "train_and_multipredict"

    trained, multipredicted, parameter = train_dataset(X, y, None, {}, classifier, parameter)

    print("\nAverage Binary Classification Performance:")

    model_list = trained["cda"]["model"]
    test_perf_list = [m["test_perf"] for m in model_list]
    test_perf_array = np.array(test_perf_list, dtype=float)
    mean_perf = np.mean(test_perf_array, axis=0)

    metrics = trained["cda"]["parameter"].get("metrics", [])
    for metric_name, perf_value in zip(metrics, mean_perf):
        print(f"{metric_name}: {perf_value}")

    if parameter["train_mode"] == "train_and_multipredict":
        print("\nAverage Multiclass Prediction Performance:")
        multi_perf_array = np.array(multipredicted["cda"]["test_perf"], dtype=float)
        mean_perf_multi = np.mean(multi_perf_array, axis=0)
        for metric_name, perf_value in zip(metrics, mean_perf_multi):
            print(f"{metric_name}: {perf_value}")


if __name__ == "__main__":
    main()
