from pathlib import Path

import tensorflow as tf
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "seg_train"
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
SEED = 42
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CLASS_NAMES = sorted(path.name for path in DATA_DIR.iterdir() if path.is_dir())
class_indices = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}

image_paths = []
image_labels = []
for class_name in CLASS_NAMES:
    for image_path in sorted((DATA_DIR / class_name).iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(str(image_path))
            image_labels.append(class_indices[class_name])

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths,
    image_labels,
    test_size=0.3,
    stratify=image_labels,
    random_state=SEED,
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.5,
    stratify=temp_labels,
    random_state=SEED,
)


def load_image(image_path, label):
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0
    return image, label


def make_dataset(paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


train_ds = make_dataset(train_paths, train_labels, training=True)
val_ds = make_dataset(val_paths, val_labels)
test_ds = make_dataset(test_paths, test_labels)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.1),
])
train_ds = train_ds.map(
    lambda images, labels: (data_augmentation(images, training=True), labels),
    num_parallel_calls=tf.data.AUTOTUNE,
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)
