from tensorflow import keras


def augmentation():
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.5),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomBrightness(0.05),
            keras.layers.RandomContrast(0.05),
            keras.layers.RandomSaturation((0.4875, 0.5125)),
        ]
    )
