from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from cnn import model
from prepare_data import test_ds, train_ds, val_ds


EPOCHS = 30
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """Save loss and accuracy curves for training and validation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history_data = history.history
    epochs = range(1, len(history_data["loss"]) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history_data["loss"], label="Training loss")
    axes[0].plot(epochs, history_data["val_loss"], label="Validation loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history_data["accuracy"], label="Training accuracy")
    axes[1].plot(epochs, history_data["val_accuracy"], label="Validation accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close(figure)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)
reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-6,
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_learning_rate],
)

plot_training_history(history)

test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
