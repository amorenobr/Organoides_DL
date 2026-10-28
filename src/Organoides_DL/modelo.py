from tensorflow import keras

from Organoides_DL.augmentation import augmentation


def construir_modelo():
    base = keras.applications.DenseNet121(
        weights="imagenet", input_shape=(224, 224, 3), include_top=False
    )
    base.trainable = False

    entradas = keras.Input(shape=(224, 224, 3))
    x = augmentation()(entradas)
    x = keras.applications.densenet.preprocess_input(x)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    salidas = keras.layers.Dense(3, activation="softmax")(x)

    modelo = keras.Model(entradas, salidas)
    modelo.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo
