
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "cnn_model.keras"
CLASS_NAMES_PATH = ROOT / "models" / "class_names.json"
DATA_DIR = ROOT / "data" / "raw" / "seg_train"

IMAGE_SIZE = (150, 150)


def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return tf.keras.models.load_model(MODEL_PATH)


def load_class_names():
    if CLASS_NAMES_PATH.exists():
        return json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))

    # Existing models were trained with image_dataset_from_directory, which
    # assigns class indices in sorted directory-name order.
    return sorted(path.name for path in DATA_DIR.iterdir() if path.is_dir())


def predict_image(image_path, model=None):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if model is None:
        model = load_trained_model()

    # Load and resize the image using the same dimensions as training.
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb",
    )

    # Convert the image to an array.
    image_array = tf.keras.utils.img_to_array(image)
    image_array = image_array / 255.0

    # Add the batch dimension: (150, 150, 3) -> (1, 150, 150, 3).
    image_array = np.expand_dims(image_array, axis=0)

    # Predict the class probabilities.
    probabilities = model.predict(image_array, verbose=0)[0]
    class_names = load_class_names()
    if len(class_names) != len(probabilities):
        raise ValueError(
            "The saved class names do not match the model output size. "
            "Retrain the model to recreate matching class metadata."
        )

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    print(f"Predicted class: {predicted_class}")
    print(f"Probability: {confidence:.2%}")

    return {
        "class": predicted_class,
        "probability": confidence,
        "probabilities": dict(zip(class_names, probabilities.tolist())),
    }



if __name__ == "__main__":
    model = load_trained_model()
    print("CNN Image Classifier ready!")
    print("Enter an image path, or type 'q' to quit.\n")

    while True:
        image_path = input("Image path: ").strip()

        if image_path.lower() == "q":
            print("Exiting prediction.")
            break

        try:
            predict_image(image_path, model=model)
            print()
        except (FileNotFoundError, ValueError) as error:
            print(f"Error: {error}\n")
