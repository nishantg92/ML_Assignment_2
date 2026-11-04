"""Streamlit interface for the Adult Income classification models."""

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logger
from src.pipeline.predict_pipeline import (
    EvaluationResult,
    MODEL_ARTIFACTS,
    evaluate_test_data,
    get_input_schema,
    load_model,
    predict_income,
)

TEST_DATA_PATH = Path(__file__).resolve().parent / "notebook" / "test_data.csv"


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


@st.cache_data(show_spinner=False, max_entries=3)
def cached_test_results(csv_bytes: bytes) -> dict[str, EvaluationResult]:
    test_data = pd.read_csv(BytesIO(csv_bytes))
    return {
        model_name: evaluate_test_data(cached_model(model_name), test_data)
        for model_name in MODEL_ARTIFACTS
    }


st.set_page_config(page_title="Adult Income Predictor", page_icon=":material/analytics:")
st.title("Adult income predictor")
st.write(
    "Compare the saved classification models on held-out test data, or estimate "
    "whether an adult's income is above $50K per year."
)


selected_model = st.sidebar.selectbox("Classification model", list(MODEL_ARTIFACTS))

st.subheader("Evaluate saved models on test data")
st.caption(
    "Use the bundled held-out test data or upload a replacement test CSV (up to 10 MB). "
    "The CSV must include all 14 input features and the income target. "
    "Models are loaded from saved artifacts; they are not retrained."
)
uploaded_test_data = st.file_uploader(
    "Upload held-out test data (CSV)",
    type=["csv"],
    max_upload_size=10,
    help="Upload test rows only, including the 14 model features and the income column.",
)

if uploaded_test_data is not None:
    test_csv_bytes = uploaded_test_data.getvalue()
    test_data_name = uploaded_test_data.name
elif TEST_DATA_PATH.is_file():
    test_csv_bytes = TEST_DATA_PATH.read_bytes()
    test_data_name = TEST_DATA_PATH.name
else:
    test_csv_bytes = None
    test_data_name = ""
    st.info("Upload a test CSV to compare the saved models.")

if test_csv_bytes is not None:
    try:
        test_results = cached_test_results(test_csv_bytes)
    except (
        CustomException,
        FileNotFoundError,
        ValueError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as exc:
        logger.exception("Could not evaluate test CSV %s", test_data_name)
        st.error(f"Could not evaluate the test CSV: {exc}")
    else:
        st.caption(f"Evaluation results for `{test_data_name}`")
        evaluated_rows = sum(map(sum, test_results[selected_model].confusion_matrix))
        st.caption(f"Rows evaluated: {evaluated_rows:,}")
        comparison = pd.DataFrame(
            [
                {"Model": model_name, **result.metrics}
                for model_name, result in test_results.items()
            ]
        ).set_index("Model")
        st.dataframe(comparison, width="stretch")

        selected_result = test_results[selected_model]
        st.subheader(f"Detailed results: {selected_model}")
        report_rows = {
            label: values if isinstance(values, dict) else {"score": values}
            for label, values in selected_result.classification_report.items()
        }
        details_column, matrix_column = st.columns(2)
        details_column.dataframe(
            pd.DataFrame.from_dict(report_rows, orient="index").rename_axis("Class"),
            width="stretch",
        )
        matrix_column.dataframe(
            pd.DataFrame(
                selected_result.confusion_matrix,
                index=["Actual <=50K", "Actual >50K"],
                columns=["Predicted <=50K", "Predicted >50K"],
            ),
            width="stretch",
        )

st.divider()
st.subheader("Try an individual prediction")

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
