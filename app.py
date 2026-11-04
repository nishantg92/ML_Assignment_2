"""Streamlit interface for the Adult Income classification models."""

import streamlit as st
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logger
from src.pipeline.predict_pipeline import (
    MODEL_ARTIFACTS,
    get_input_schema,
    load_model,
    predict_income,
)


NUMERIC_DEFAULTS = {
    "age": 39,
    "fnlwgt": 189778,
    "education.num": 10,
    "capital.gain": 0,
    "capital.loss": 0,
    "hours.per.week": 40,
}

CATEGORICAL_DEFAULTS = {
    "workclass": "Private",
    "education": "HS-grad",
    "marital.status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "native.country": "United-States",
}


@st.cache_resource
def cached_model(model_name: str) -> Pipeline:
    return load_model(model_name)


st.set_page_config(page_title="Adult Income Predictor", page_icon=":material/analytics:")
st.title("Adult income predictor")
st.write(
    "Explore the saved classification models and estimate whether an adult's "
    "income is above $50K per year."
)
st.caption(
    "This model is for educational use only. Predictions may reflect limitations "
    "or biases in the historical dataset and should not be used for decisions about people."
)

selected_model = st.sidebar.selectbox("Classification model", list(MODEL_ARTIFACTS))

try:
    model = cached_model(selected_model)
    numeric_features, categorical_features = get_input_schema(model)
except (CustomException, FileNotFoundError, ValueError):
    logger.exception("The app could not prepare model %s", selected_model)
    st.error(
        "The selected model could not be loaded. Check the application logs and "
        "confirm the model files are present."
    )
    st.stop()

st.subheader("Enter profile details")
with st.form("income_prediction"):
    features: dict[str, int | float | str] = {}
    left_column, right_column = st.columns(2)

    for index, feature in enumerate(numeric_features):
        column = left_column if index % 2 == 0 else right_column
        step = 100 if feature == "fnlwgt" else 1
        features[feature] = column.number_input(
            feature.replace(".", " ").replace("_", " ").capitalize(),
            value=NUMERIC_DEFAULTS.get(feature, 0),
            step=step,
            key=f"numeric_{feature}",
        )

    for index, (feature, options) in enumerate(categorical_features.items()):
        column = left_column if index % 2 == 0 else right_column
        default = CATEGORICAL_DEFAULTS.get(feature, options[0])
        selected_index = options.index(default) if default in options else 0
        features[feature] = column.selectbox(
            feature.replace(".", " ").replace("_", " ").capitalize(),
            options,
            index=selected_index,
            key=f"categorical_{feature}",
        )

    submitted = st.form_submit_button("Predict income", type="primary")

if submitted:
    try:
        prediction = predict_income(model, features)
    except (CustomException, ValueError):
        logger.exception("Income prediction failed")
        st.error("Prediction failed. Check the application logs for details.")
    else:
        st.success(f"Predicted income category: {prediction.income}")
        st.metric(
            "Estimated probability of income above $50K",
            f"{prediction.probability_over_50k:.1%}",
        )
