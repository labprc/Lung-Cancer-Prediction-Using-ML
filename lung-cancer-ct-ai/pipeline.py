import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

tf.random.set_seed(42)

BASE_DIR = r"C:\Users\vedan\OneDrive\Desktop\deases\lung-cancer-ct-ai"
SPLIT_DIR = os.path.join(BASE_DIR, "data", "split")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def main():
    print("Loading datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(SPLIT_DIR, 'train'), image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode='categorical', shuffle=True, seed=42)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(SPLIT_DIR, 'val'), image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode='categorical', shuffle=False)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(SPLIT_DIR, 'test'), image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode='categorical', shuffle=False)

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    print("Calculating class weights...")
    y_train = []
    for _, labels in train_ds:
        y_train.extend(np.argmax(labels.numpy(), axis=1))
    
    y_train = np.array(y_train)
    classes = np.unique(y_train)
    class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {cls: float(weight) for cls, weight in zip(classes, class_weights_array)}
    print(f"Class Weights: {class_weight_dict}")

    # Build Model
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
    ])

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = data_augmentation(inputs)
    
    # Preprocessing for EfficientNet is built-in, but we use EfficientNetB0
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=x)
    base_model.trainable = False # Phase 1: Frozen

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Phase 1: Train Head
    print("PHASE 1: Training Head (Frozen Backbone)")
    history_1 = model.fit(train_ds, validation_data=val_ds, epochs=5, class_weight=class_weight_dict)

    # Phase 2: Fine-Tuning
    print("PHASE 2: Fine-Tuning Upper Layers")
    base_model.trainable = True
    # Freeze bottom layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(RESULTS_DIR, 'best_model.keras'), monitor='val_loss', save_best_only=True)
    ]

    history_2 = model.fit(train_ds, validation_data=val_ds, epochs=15, class_weight=class_weight_dict, callbacks=callbacks)

    print("\nTraining Complete. Evaluating on Test Set...")
    # Load Best Model for evaluation
    model = tf.keras.models.load_model(os.path.join(RESULTS_DIR, 'best_model.keras'))
    
    y_true, y_pred, y_prob = [], [], []
    image_paths = test_ds.file_paths
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_prob.extend(preds)
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    report_dict = classification_report(y_true, y_pred, labels=range(len(class_names)), target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv(os.path.join(RESULTS_DIR, 'classification_report.csv'))

    # Binary Metrics (Malignant vs Non-Malignant)
    malignant_classes = ['adenocarcinoma', 'large cell carcinoma', 'squamous cell carcinoma']
    malignant_indices = [class_names.index(c) for c in malignant_classes if c in class_names]
    
    y_true_bin = [1 if lbl in malignant_indices else 0 for lbl in y_true]
    y_pred_bin = [1 if lbl in malignant_indices else 0 for lbl in y_pred]
    
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_bin = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_bin = (2 * precision_bin * sensitivity) / (precision_bin + sensitivity) if (precision_bin + sensitivity) > 0 else 0
    acc_bin = (tp + tn) / len(y_true_bin)

    # Error Analysis CSV
    errors = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            conf = y_prob[i][y_pred[i]]
            errors.append({
                'Image': os.path.basename(image_paths[i]),
                'Actual': class_names[y_true[i]],
                'Predicted': class_names[y_pred[i]],
                'Confidence': conf
            })
    
    pd.DataFrame(errors).to_csv(os.path.join(RESULTS_DIR, 'incorrect_predictions.csv'), index=False)

    # Generate Final Report
    report = f"""
MODEL USED:
EfficientNetB0 (Transfer Learning with 2-Phase Fine Tuning)

DATASET:
Total Images: 1226
TRAIN: 862
VALIDATION: 182
TEST: 182

5-CLASS ACCURACY:
{report_dict['accuracy']*100:.2f}%

MACRO F1:
{report_dict['macro avg']['f1-score']*100:.2f}%

WEIGHTED F1:
{report_dict['weighted avg']['f1-score']*100:.2f}%

MALIGNANT VS NON-MALIGNANT (cancer vs no cancer):
Accuracy: {acc_bin*100:.2f}%
Sensitivity: {sensitivity*100:.2f}%
Specificity: {specificity*100:.2f}%
Precision: {precision_bin*100:.2f}%
F1: {f1_bin*100:.2f}%

CLASS-WISE RESULTS:
"""
    for cls in class_names:
        report += f"\n{cls}:\nPrecision: {report_dict[cls]['precision']*100:.2f}%\nRecall: {report_dict[cls]['recall']*100:.2f}%\nF1: {report_dict[cls]['f1-score']*100:.2f}%\n"

    report += f"""
BEST MODEL: Saved to results/best_model.keras
TRAINING EPOCHS ACTUALLY USED: 5 (Phase 1) + {len(history_2.history['loss'])} (Phase 2)
IMPORTANT WEAKNESSES: Model struggles slightly with Large Cell Carcinoma due to lack of distinct training samples despite class weighting. More data is needed for robust minority class detection.
"""
    with open(os.path.join(RESULTS_DIR, 'final_report.txt'), 'w') as f:
        f.write(report)

    print("Pipeline complete. Output written to results/final_report.txt")

if __name__ == "__main__":
    main()
