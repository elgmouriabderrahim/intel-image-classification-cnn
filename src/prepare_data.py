import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "seg_train"

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=(150, 150),
    batch_size=32,
    shuffle=True,
    validation_split=0.3,
    subset="training",
)

temp_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=(150, 150),
    batch_size=32,
    shuffle=True,
    validation_split=0.3,
    subset="validation",
)
val_ds = temp_ds.take(len(temp_ds) // 2)
test_ds = temp_ds.skip(len(temp_ds) // 2)


normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda images, labels: (normalization_layer(images), labels))
val_ds = val_ds.map(lambda images, labels: (normalization_layer(images), labels))
test_ds = test_ds.map(lambda images, labels: (normalization_layer(images), labels))


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.1),
])
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
)