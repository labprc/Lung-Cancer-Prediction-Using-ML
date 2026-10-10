import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

IMAGE_ROOT = r"C:\Users\vedan\OneDrive\Desktop\deases\lung-cancer-ct-ai\data\split\test"
MODEL_PATH = "results/best_model.keras"
CLASS_NAMES_PATH = "models/class_names.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def evaluate_malignant_sensitivity():

    print("Loading model and class names...")

    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return

    if not os.path.exists(CLASS_NAMES_PATH):
        print("Class names not found.")
        return

    if not os.path.exists(IMAGE_ROOT):
        print("Test dataset not found.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    print(f"Loaded classes: {class_names}")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        IMAGE_ROOT,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    y_true = []
    y_pred = []

    print("\nRunning predictions on test dataset...")

    for images, labels in test_ds:

        predictions = model.predict(
            images,
            verbose=0
        )

        y_pred.extend(
            np.argmax(predictions, axis=1)
        )

        y_true.extend(
            np.argmax(labels.numpy(), axis=1)
        )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n" + "=" * 60)
    print(" MEDICAL AI EVALUATION REPORT")
    print("=" * 60)

    print(f"\nTotal Test Images: {len(y_true)}")

    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    malignant_classes = [
        "adenocarcinoma",
        "large cell carcinoma",
        "squamous cell carcinoma"
    ]

    malignant_indices = [
        class_names.index(c)
        for c in malignant_classes
        if c in class_names
    ]

    y_true_binary = np.array([
        1 if label in malignant_indices else 0
        for label in y_true
    ])

    y_pred_binary = np.array([
        1 if label in malignant_indices else 0
        for label in y_pred
    ])

    tn, fp, fn, tp = confusion_matrix(
        y_true_binary,
        y_pred_binary,
        labels=[0, 1]
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    f1_score = (
        2 * precision * sensitivity /
        (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    accuracy = np.mean(y_true == y_pred)

    print(f"\nOVERALL 5-CLASS ACCURACY : {accuracy * 100:.2f}%")

    print("\n" + "=" * 60)
    print(" MALIGNANT VS NON-MALIGNANT")
    print("=" * 60)

    print("\nBINARY CONFUSION MATRIX")
    print("-" * 60)

    print(
        f"{'':25}"
        f"{'Predicted Non-Malig':22}"
        f"{'Predicted Malig':20}"
    )

    print(
        f"{'Actual Non-Malig':25}"
        f"{tn:<22}"
        f"{fp:<20}"
    )

    print(
        f"{'Actual Malig':25}"
        f"{fn:<22}"
        f"{tp:<20}"
    )

    print("\nCLINICAL METRICS")
    print("-" * 60)

    print(f"Total Malignant Cases : {tp + fn}")
    print(f"False Negatives       : {fn}")
    print(f"False Positives       : {fp}")
    print(f"Sensitivity (Recall)  : {sensitivity * 100:.2f}%")
    print(f"Specificity           : {specificity * 100:.2f}%")
    print(f"Precision             : {precision * 100:.2f}%")
    print(f"F1-Score              : {f1_score * 100:.2f}%")

    print("\n" + "=" * 60)
    print(" 5-CLASS BREAKDOWN")
    print("=" * 60)

    for cls in class_names:

        metrics = report[cls]

        print(f"\n[{cls.upper()}]")
        print(f"Recall    : {metrics['recall'] * 100:.2f}%")
        print(f"Precision : {metrics['precision'] * 100:.2f}%")
        print(f"F1-Score  : {metrics['f1-score'] * 100:.2f}%")
        print(f"Samples   : {metrics['support']}")

    print("\n" + "=" * 60)
    print(" TEST DATA DISTRIBUTION")
    print("=" * 60)

    for i, cls in enumerate(class_names):

        actual_count = np.sum(y_true == i)
        predicted_count = np.sum(y_pred == i)

        print(
            f"{cls}: "
            f"Actual={actual_count}, "
            f"Predicted={predicted_count}"
        )

    print("\n" + "=" * 60)
    print(" 5-CLASS CONFUSION MATRIX")
    print("=" * 60)

    cm_5 = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(class_names))
    )

    print("\nActual (Rows) × Predicted (Columns):\n")
    print(cm_5)

    print("\nClass Order:")
    for i, cls in enumerate(class_names):
        print(f"{i}: {cls}")

    print("\n" + "=" * 60)
    print(" EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_malignant_sensitivity()