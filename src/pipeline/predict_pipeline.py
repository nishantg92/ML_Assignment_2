import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logger


MODEL_ARTIFACTS = {
    "Logistic regression": "logistic_regression.joblib",
    "Decision tree": "decision_tree.joblib",
    "K-nearest neighbors": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support vector machine": "support_vector_machine.joblib",
}

MODEL_DIR = Path(__file__).resolve().parents[2] / "notebook" / "model"
INCOME_LABELS = {0: "<=50K", 1: ">50K"}


@dataclass(frozen=True)
class Prediction:
    income: str
    probability_over_50k: float


def load_model(model_name: str) -> Pipeline:
    try:
        artifact_name = MODEL_ARTIFACTS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name}") from exc

    model_path = MODEL_DIR / artifact_name
    if not model_path.is_file():
        error = FileNotFoundError(f"Model artifact not found: {model_path}")
        logger.error("%s", error)
        raise error

    try:
        model = joblib.load(model_path)
        if not isinstance(model, Pipeline):
            raise TypeError(f"Model artifact is not a scikit-learn Pipeline: {model_path}")
    except Exception as exc:
        logger.exception("Could not load model artifact %s", model_path)
        raise CustomException("Could not load the selected model", sys) from exc

    logger.info("Loaded model artifact %s", model_path.name)
    return model


def get_input_schema(
    model: Pipeline,
) -> tuple[list[str], dict[str, tuple[str, ...]]]:
    try:
        preprocessor = model.named_steps["preprocessor"]
        numeric_features: list[str] = []
        categorical_features: dict[str, tuple[str, ...]] = {}

        for name, transformer, columns in preprocessor.transformers_:
            if name == "num":
                numeric_features = list(columns)
            elif name == "cat":
                categories = transformer.categories_
                categorical_features = {
                    column: tuple(str(value) for value in values)
                    for column, values in zip(columns, categories, strict=True)
                }

        if not numeric_features or not categorical_features:
            raise ValueError("The model does not contain the expected input schema")

        return numeric_features, categorical_features
    except Exception as exc:
        logger.exception("Could not read the selected model's input schema")
        raise CustomException("Could not read the model input schema", sys) from exc


def predict_income(model: Pipeline, features: Mapping[str, object]) -> Prediction:
    expected_features = list(model.feature_names_in_)
    missing_features = set(expected_features) - features.keys()
    extra_features = features.keys() - set(expected_features)
    if missing_features or extra_features:
        raise ValueError(
            "Input fields do not match the model schema "
            f"(missing: {sorted(missing_features)}, extra: {sorted(extra_features)})"
        )

    input_data = pd.DataFrame([features], columns=expected_features)
    try:
        predicted_class = int(model.predict(input_data)[0])
        classifier = model.named_steps["classifier"]
        classes = list(classifier.classes_)
        if 1 not in classes:
            raise ValueError("The model does not contain the >50K income class")
        probability = float(model.predict_proba(input_data)[0, classes.index(1)])
        income = INCOME_LABELS[predicted_class]
    except Exception as exc:
        logger.exception("Prediction failed")
        raise CustomException("The selected model could not make a prediction", sys) from exc

    logger.info("Prediction completed")
    return Prediction(income=income, probability_over_50k=probability)