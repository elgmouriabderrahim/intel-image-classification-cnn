# Intel Image Classification CNN

An image classifier built with TensorFlow/Keras for the Intel Image Classification dataset. The project includes data preparation, stratified train/validation/test splits, CNN training, evaluation notebooks, command-line prediction, and a Streamlit dashboard.

The model classifies images into six scene categories: buildings, forest, glacier, mountain, sea, and street. Its probability output is a confidence score, not a guarantee that a prediction is correct.

## Project layout

```text
dashboard/app.py       Streamlit prediction dashboard
src/prepare_data.py    Stratified data splits and preprocessing
src/cnn.py             CNN model definition
src/train.py           Model training and test evaluation
src/predict.py         Command-line and shared prediction code
models/                Trained model and class-name metadata
notebooks/             Exploration, preprocessing, evaluation, experiments
```

## Requirements

- Python 3.11
- Dependencies from `requirements.txt`
- The Intel Image Classification training images at `data/raw/seg_train/`, arranged in one subfolder per class
- A trained model at `models/cnn_model.keras` and matching `models/class_names.json`

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the dashboard locally

From the project root:

```bash
streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`), upload a JPG or PNG, and select **Classify image**.

## Run with Docker

Install and start Docker with the Compose plugin, then run from the project root:

```bash
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501). Compose builds the image and mounts the host `models/` directory read-only, so the dashboard uses the project’s trained model and class metadata. Stop it with `Ctrl+C`, or run `docker compose down` in another terminal.

After retraining the model on your host, restart the dashboard container so its cached model reloads:

```bash
docker compose restart dashboard
```

## Train the model

Put the training images under `data/raw/seg_train/<class-name>/`, then run:

```bash
python src/train.py
```

Training writes `models/cnn_model.keras`, `models/class_names.json`, and `models/training_curves.png`. The training script evaluates the resulting model on the held-out test split.

## Predict from the command line

```bash
python src/predict.py
```

Enter an image path when prompted. Type `q` to quit.

## Dependencies

Install the dependencies listed in [`requirements.txt`](requirements.txt). Docker uses the same list when building the dashboard image.
