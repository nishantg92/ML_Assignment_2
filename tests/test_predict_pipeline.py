import pytest

from src.pipeline.predict_pipeline import (
    MODEL_ARTIFACTS,
    get_input_schema,
    load_model,
    predict_income,
)


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
