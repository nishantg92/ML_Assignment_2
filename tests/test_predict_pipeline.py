from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.predict_pipeline import (
    MODEL_ARTIFACTS,
    evaluate_test_data,
    get_input_schema,
    load_model,
    predict_income,
)

TEST_DATA_PATH = Path(__file__).resolve().parents[1] / "notebook" / "test_data.csv"


def test_saved_models_are_available():
    for model_name in MODEL_ARTIFACTS:
        assert load_model(model_name) is not None


def test_logistic_regression_predicts_from_model_schema():
    model = load_model("Logistic regression")
    numeric_features, categorical_features = get_input_schema(model)

    profile = {feature: 0 for feature in numeric_features}
    profile.update(
        {
            feature: options[0]
            for feature, options in categorical_features.items()
        }
    )
    result = predict_income(model, profile)

    assert result.income in {"<=50K", ">50K"}
    assert 0.0 <= result.probability_over_50k <= 1.0


def test_prediction_rejects_missing_fields():
    model = load_model("Logistic regression")

    with pytest.raises(ValueError, match="Input fields do not match"):
        predict_income(model, {})


def test_load_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown model"):
        load_model("unknown")


@pytest.mark.parametrize("model_name", MODEL_ARTIFACTS)
def test_evaluate_saved_models_on_held_out_test_data(model_name):
    test_data = pd.read_csv(TEST_DATA_PATH, nrows=100)
    result = evaluate_test_data(load_model(model_name), test_data)

    assert set(result.metrics) == {
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
    }
    assert all(0.0 <= result.metrics[name] <= 1.0 for name in ("Accuracy", "AUC"))
    assert sum(map(sum, result.confusion_matrix)) == len(test_data)
    assert "<=50K" in result.classification_report
    assert ">50K" in result.classification_report


def test_evaluate_test_data_rejects_missing_required_columns():
    test_data = pd.read_csv(TEST_DATA_PATH, nrows=10).drop(columns=["age"])

    with pytest.raises(ValueError, match="missing required columns"):
        evaluate_test_data(load_model("Logistic regression"), test_data)
