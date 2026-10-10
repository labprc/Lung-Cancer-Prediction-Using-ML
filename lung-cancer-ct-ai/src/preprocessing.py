import tensorflow as tf

def get_data_augmentation(enable=True):
    if not enable:
        return tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255)
        ])

    return tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ])


def preprocess_image(image_path, image_size=(224, 224)):
    img = tf.keras.utils.load_img(
        image_path,
        target_size=image_size
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)

    return img_array, img