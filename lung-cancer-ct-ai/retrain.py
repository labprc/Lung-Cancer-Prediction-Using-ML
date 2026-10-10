import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import json

# ============================================================
# REAL CT DATASET — LIDC-IDRI (TCIA)
# ============================================================

DATASET_NAME = "LIDC-IDRI"
DATASET_URL = "https://www.cancerimagingarchive.net/collection/lidc-idri/"

# Local directory where the downloaded real CT data will be stored
DATASET_ROOT = r"D:\LungCancerAI\LIDC-IDRI"

# After downloading and organizing the dataset:
IMAGE_ROOT = r"C:\Users\vedan\OneDrive\Desktop\deases\lung-cancer-ct-ai\data\train"

# Automatically create a 15% final test split
TEST_SIZE = 0.15

# Remaining 85% is used for training + validation
TRAIN_VALID_SIZE = 0.85

RANDOM_STATE = 42

# Config
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5 

def retrain_model():
    print(f"Checking dataset at: {IMAGE_ROOT}")
    if not os.path.exists(IMAGE_ROOT):
        print(f"ERROR: The directory {IMAGE_ROOT} does not exist.")
        print("Please ensure your dataset is organized in subfolders by class inside this directory.")
        return

    print(f"Loading dataset with a {TEST_SIZE*100}% test/validation split...")
    
    # 85% Training + Validation
    train_ds = tf.keras.utils.image_dataset_from_directory(
        IMAGE_ROOT,
        validation_split=TEST_SIZE,
        subset="training",
        seed=RANDOM_STATE,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    # 15% Final Test Split
    test_ds = tf.keras.utils.image_dataset_from_directory(
        IMAGE_ROOT,
        validation_split=TEST_SIZE,
        subset="validation",
        seed=RANDOM_STATE,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    if num_classes == 0:
        print("No classes found. Ensure there are subfolders in the IMAGE_ROOT.")
        return

    # Data Augmentation to prevent overfitting
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # Build Model
    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = data_augmentation(inputs)
    
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=x)
    base_model.trainable = True # Unfreeze the base model for better accuracy

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight

    print("\nCalculating Class Distribution and Weights...")
    y_train = []
    for _, labels in train_ds:
        y_train.extend(np.argmax(labels.numpy(), axis=1))
    
    y_train = np.array(y_train)
    classes = np.unique(y_train)
    
    print("\nTRAIN CLASS DISTRIBUTION:")
    for i, cls_name in enumerate(class_names):
        count = np.sum(y_train == i)
        print(f"{cls_name}: {count}")

    class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights_array)}
    print(f"\nComputed Class Weights to handle imbalance:\n{class_weight_dict}\n")

    print("Starting training on the LIDC-IDRI dataset with class weights...")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    # Save the new model
    os.makedirs('models', exist_ok=True)
    model.save('models/best_model.keras')
    
    with open('models/class_names.json', 'w') as f:
        json.dump(class_names, f, indent=4)

    print("\nTraining Complete! The new model has been saved.")

if __name__ == "__main__":
    retrain_model()
