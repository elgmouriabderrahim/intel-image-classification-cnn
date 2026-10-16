import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models


model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),

    layers.Conv2D(32, kernel_size=3, padding="same"),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=2),

    layers.Conv2D(64, kernel_size=3, padding="same"),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=2),

    layers.Conv2D(128, kernel_size=3, padding="same"),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=2),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(6, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()
