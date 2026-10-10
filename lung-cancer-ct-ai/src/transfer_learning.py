import tensorflow as tf


def build_transfer_model(
    model_name,
    input_shape,
    num_classes,
    fine_tune=False,
    fine_tune_layers=30
):
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    if model_name == "EfficientNetB0":
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs
        )
    elif model_name == "ResNet50":
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs
        )
    elif model_name == "MobileNetV2":
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs
        )
    else:
        raise ValueError(
            f"Model {model_name} not supported."
        )

    base_model.trainable = False

    if fine_tune:
        base_model.trainable = True

        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False

        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    x = base_model.output

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    x = tf.keras.layers.Dropout(0.25)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="prediction"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"{model_name}_LungCancer"
    )

    return model


def unfreeze_for_fine_tuning(model, fine_tune_layers=30):
    base_model = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Pretrained backbone not found.")

    base_model.trainable = True

    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    return model