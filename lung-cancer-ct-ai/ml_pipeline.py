import os
import cv2
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

SPLIT_DIR = r"C:\Users\vedan\OneDrive\Desktop\deases\lung-cancer-ct-ai\data\split"
RESULTS_DIR = r"C:\Users\vedan\OneDrive\Desktop\deases\lung-cancer-ct-ai\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# For traditional ML, we reduce image size to avoid exploding feature space
IMG_SIZE = 64 

def load_data(split):
    X, y = [], []
    folder = os.path.join(SPLIT_DIR, split)
    classes = sorted(os.listdir(folder))
    for i, cls in enumerate(classes):
        cls_folder = os.path.join(folder, cls)
        if not os.path.isdir(cls_folder):
            continue
        for img_name in os.listdir(cls_folder):
            img_path = os.path.join(cls_folder, img_name)
            # Read image in Grayscale to simplify features for traditional ML
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                # Flatten the 2D image into a 1D feature array (64x64 = 4096 features)
                X.append(img.flatten())
                y.append(i)
    return np.array(X), np.array(y), classes

def main():
    print("="*50)
    print(" TRADITIONAL MACHINE LEARNING PIPELINE ")
    print("="*50)
    
    print("Loading training data...")
    X_train, y_train, classes = load_data('train')
    print("Loading test data...")
    X_test, y_test, _ = load_data('test')

    print(f"Features per image: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Handle Class Imbalance
    class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights_array)}
    
    print("\nTraining Random Forest Classifier (No Deep Learning/AI)...")
    # Using Random Forest, a pure traditional Machine Learning algorithm
    clf = RandomForestClassifier(n_estimators=300, class_weight=class_weight_dict, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("\nEvaluating on untouched Test Set...")
    y_pred = clf.predict(X_test)
    
    # 5-Class Evaluation
    print("\n--- 5-CLASS EVALUATION ---")
    report_dict = classification_report(y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))
    
    # Binary Evaluation (Malignant vs Non-Malignant)
    malignant_classes = ['adenocarcinoma', 'large cell carcinoma', 'squamous cell carcinoma']
    malignant_indices = [classes.index(c) for c in malignant_classes if c in classes]
    
    y_true_bin = [1 if lbl in malignant_indices else 0 for lbl in y_test]
    y_pred_bin = [1 if lbl in malignant_indices else 0 for lbl in y_pred]
    
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_bin = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_bin = (2 * precision_bin * sensitivity) / (precision_bin + sensitivity) if (precision_bin + sensitivity) > 0 else 0
    acc_bin = (tp + tn) / len(y_true_bin)
    
    print("\n--- MALIGNANT VS NON-MALIGNANT (CANCER VS NO CANCER) ---")
    print(f"Accuracy:    {acc_bin*100:.2f}%")
    print(f"Sensitivity: {sensitivity*100:.2f}%")
    print(f"Specificity: {specificity*100:.2f}%")
    print(f"Precision:   {precision_bin*100:.2f}%")
    print(f"F1-Score:    {f1_bin*100:.2f}%")

    # Save Model
    model_path = os.path.join(RESULTS_DIR, 'traditional_ml_model.pkl')
    joblib.dump({'model': clf, 'classes': classes, 'img_size': IMG_SIZE}, model_path)
    print(f"\nTraditional ML Model saved to {model_path}")

if __name__ == "__main__":
    main()
