import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from src.config import MODELS_DIR, OUTPUT_DIR


def save_json(data, filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    filepath = os.path.join(MODELS_DIR, filename)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def load_json(filename):
    filepath = os.path.join(MODELS_DIR, filename)

    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)

    return None


def plot_training_history(history, model_name="model"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Training Accuracy", linewidth=2)

    if len(val_acc) > 0:
        plt.plot(
            epochs,
            val_acc,
            label="Validation Accuracy",
            linewidth=2
        )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Training Loss", linewidth=2)

    if len(val_loss) > 0:
        plt.plot(
            epochs,
            val_loss,
            label="Validation Loss",
            linewidth=2
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{model_name}_training_history.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_confusion_matrix(cm, class_names, model_name="model"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5
    )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f"Confusion Matrix - {model_name}")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{model_name}_confusion_matrix.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def calculate_metrics(y_true, y_pred, class_names):
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"])
    }

    return metrics


def save_metrics(metrics, model_name="model"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filepath = os.path.join(
        OUTPUT_DIR,
        f"{model_name}_metrics.json"
    )

    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=4)