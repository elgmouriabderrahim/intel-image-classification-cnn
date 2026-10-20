from pathlib import Path
import sys
import tempfile

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict import load_trained_model, predict_image


st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="",
    layout="wide",
)


@st.cache_resource
def get_model():
    return load_trained_model()


st.title("CNN Image Classifier")
st.write("Upload an image to see which scene class the model predicts.")

st.sidebar.header("About")
st.sidebar.write(
    "Classifies images into buildings, forest, glacier, mountain, sea, or street."
)
st.sidebar.caption(
    "The probability is the model’s confidence score, not a guarantee that the prediction is correct."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG or PNG image to classify.",
)

if uploaded_file is not None:
    image_col, result_col = st.columns([1, 1], gap="large")

    with image_col:
        st.subheader("Image")
        st.image(uploaded_file, use_container_width=True)

    with result_col:
        st.subheader("Prediction")
        if st.button("Classify image", type="primary", use_container_width=True):
            temp_path = None
            try:
                suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = Path(temp_file.name)

                model = get_model()
                result = predict_image(temp_path, model=model)

                st.success(f"Predicted class: **{result['class'].title()}**")
                st.metric("Model confidence", f"{result['probability']:.2%}")

                probabilities = pd.DataFrame(
                    {
                        "Class": [name.title() for name in result["probabilities"]],
                        "Probability": list(result["probabilities"].values()),
                    }
                ).set_index("Class")
                st.caption("Scores for all classes")
                st.bar_chart(probabilities, x_label="Class", y_label="Probability")
            except (FileNotFoundError, ValueError) as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"Could not classify this image: {error}")
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
else:
    st.info("Upload a JPG or PNG image to get started.")
