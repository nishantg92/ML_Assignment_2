import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
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


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    confusion_matrix: list[list[int]]
    classification_report: dict[str, object]


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


def evaluate_test_data(
    model: Pipeline, test_data: pd.DataFrame, target_column: str = "income"
) -> EvaluationResult:
    expected_features = list(model.feature_names_in_)
    missing_columns = set(expected_features + [target_column]) - set(test_data.columns)
    if missing_columns:
        raise ValueError(
            f"Test CSV is missing required columns: {sorted(missing_columns)}"
        )
    if test_data.empty:
        raise ValueError("Test CSV must contain at least one row")

    def encode_target(value: object) -> int | None:
        if pd.isna(value):
            return None
        label = str(value).strip().rstrip(".")
        if label in {"0", "<=50K"}:
            return 0
        if label in {"1", ">50K"}:
            return 1
        return None

    y_true = test_data[target_column].map(encode_target)
    if y_true.isna().any():
        invalid_labels = test_data.loc[y_true.isna(), target_column].unique().tolist()
        raise ValueError(
            f"Target column '{target_column}' contains missing or unsupported labels: "
            f"{invalid_labels[:5]}"
        )
    y_true = y_true.astype(int)
    if set(y_true.unique()) != {0, 1}:
        raise ValueError(
            f"Target column '{target_column}' must contain both income classes "
            "'<=50K' and '>50K' to calculate all evaluation metrics"
        )

    input_data = test_data.loc[:, expected_features]
    try:
        y_pred = model.predict(input_data)
        classifier = model.named_steps["classifier"]
        classes = list(classifier.classes_)
        if 1 not in classes:
            raise ValueError("The model does not contain the >50K income class")
        y_score = model.predict_proba(input_data)[:, classes.index(1)]
    except Exception as exc:
        logger.exception("Evaluation failed")
        raise CustomException("The selected model could not evaluate test data", sys) from exc

    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_score)),
        "Precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "Recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[INCOME_LABELS[0], INCOME_LABELS[1]],
        output_dict=True,
        zero_division=0,
    )
    logger.info("Test-data evaluation completed for %d rows", len(test_data))
    return EvaluationResult(
        metrics=metrics,
        confusion_matrix=matrix,
        classification_report=report,
    )