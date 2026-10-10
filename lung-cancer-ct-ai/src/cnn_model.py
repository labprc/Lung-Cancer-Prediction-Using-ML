import tensorflow as tf

def build_custom_cnn(input_shape, num_classes):
    """
    Builds a custom CNN model dynamically adapting to num_classes.
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    if num_classes == 2:
        # We can still use softmax with 2 units for categorical crossentropy since we loaded label_mode='categorical'
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Custom_CNN")
    return model
