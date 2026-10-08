"""Classification, fairness and straggler metrics with explicit denominators."""

import numpy as np


def classification_metrics(confusion_matrix, class_names=None):
    """Rows: true labels; columns: predictions. Undefined ratios are zero.

    Macro metrics include all configured classes, including absent classes.
    Per-class accuracy means recall (correct / true support).
    """
    cm = np.asarray(confusion_matrix, dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or (cm < 0).any():
        raise ValueError("Expected a square, nonnegative confusion matrix")
    support, predicted, tp = cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
    n = int(support.sum())
    if n == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    precision = np.divide(tp, predicted, out=np.zeros(len(cm)), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(len(cm)), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(cm)), where=(precision + recall) != 0)
    names = class_names if class_names is not None else [str(i) for i in range(len(cm))]
    if len(names) != len(cm):
        raise ValueError("class_names must match the number of classes")
    result = {
        "num_samples": n, "accuracy": float(tp.sum() / n),
        "confusion_matrix": cm.tolist(), "class_names": list(names),
        "macro_average_policy": "all_configured_classes; zero_division=0",
        "per_class": [
            {"class_id": i, "class_name": names[i], "support": int(support[i]),
             "predicted_count": int(predicted[i]), "correct": int(tp[i]),
             "accuracy": float(recall[i]), "precision": float(precision[i]),
             "recall": float(recall[i]), "f1": float(f1[i])}
            for i in range(len(cm))
        ],
    }
    for name, values in [("precision", precision), ("recall", recall), ("f1", f1)]:
        result[f"{name}_macro"] = float(values.mean())
        result[f"{name}_weighted"] = float(np.dot(values, support) / n)
    return result


def client_fairness(reports):
    """Unweighted client statistics, distinct from pooled sample metrics."""
    accuracies = np.asarray([r["accuracy"] for r in reports], dtype=float)
    f1s = np.asarray([r["f1_macro"] for r in reports], dtype=float)
    return {
        "evaluated_clients": len(reports),
        "mean_client_accuracy": float(accuracies.mean()) if len(reports) else None,
        "client_accuracy_variance": float(accuracies.var(ddof=0)) if len(reports) else None,
        "worst_client_accuracy": float(accuracies.min()) if len(reports) else None,
        "mean_client_f1_macro": float(f1s.mean()) if len(reports) else None,
        "worst_client_f1_macro": float(f1s.min()) if len(reports) else None,
    }


def straggler_metrics(times):
    """Compute-only estimates; actual waiting needs transport tracing."""
    times = np.asarray(times, dtype=float)
    if not len(times):
        return {key: None for key in (
            "slowest_client_train_seconds", "median_client_train_seconds",
            "straggler_overhead_seconds", "estimated_wait_seconds_sum")}
    slowest, median = float(times.max()), float(np.median(times))
    return {"slowest_client_train_seconds": slowest,
            "median_client_train_seconds": median,
            "straggler_overhead_seconds": slowest - median,
            "estimated_wait_seconds_sum": float((slowest - times).sum())}


class FLMetricsRecorder:
    """Record round metrics, retaining unavailable values as null in JSON."""
    def __init__(self):
        self.rows = []

    def record(self, round_number, **metrics):
        if "round" in metrics:
            raise ValueError("round is supplied separately")
        import json
        row = {"round": round_number, **metrics}
        json.dumps(row, allow_nan=False)
        self.rows.append(row)

    def export(self, directory):
        import csv
        import json
        from pathlib import Path
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metrics.json").write_text(json.dumps(self.rows, indent=2, allow_nan=False), encoding="utf-8")
        fields = list(dict.fromkeys(key for row in self.rows for key in row)) or ["round"]
        with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)

    def plot(self, metric, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path
        points = [(row["round"], row[metric]) for row in self.rows if row.get(metric) is not None]
        figure, axis = plt.subplots()
        axis.plot([p[0] for p in points], [p[1] for p in points], marker="o")
        axis.set(xlabel="Round", ylabel=metric)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, bbox_inches="tight")
        plt.close(figure)
