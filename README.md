# Adult Income Classification

## a. Problem statement

Build and compare five machine-learning classification models that predict
whether an adult's income is `<=50K` or `>50K` using the features in the
Adult Income dataset. The models will be evaluated with the same held-out test
data and evaluation metrics.

## b. Dataset description

The project uses `adult.csv`, a tabular dataset containing demographic and
employment-related features. The target column is `income`; the remaining
columns are used as model features. The dataset includes both numeric and
categorical values, and `?` is used for some missing values.

The data file currently in this repository is the source dataset. Data
preprocessing, missing-value handling, and the train/test split will be
documented here once finalized.

## c. GitHub Repository Link

[https://github.com/nishantg92/ML_Assignment_2](https://github.com/nishantg92/ML_Assignment_2)

## d. Models used and evaluation

All five models will be trained and evaluated using a consistent data split.
The positive class, probability/decision-score handling for AUC, and any
preprocessing choices will be documented after implementation.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |

| Logistic Regression | TBD | TBD | TBD | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD | TBD | TBD | TBD |
| KNN | TBD | TBD | TBD | TBD | TBD | TBD |
| Naive Bayes | TBD | TBD | TBD | TBD | TBD | TBD |
| Support Vector Machine | TBD | TBD | TBD | TBD | TBD | TBD |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | To be completed after training and evaluation. |
| Decision Tree | To be completed after training and evaluation. |
| KNN | To be completed after training and evaluation. |
| Naive Bayes | To be completed after training and evaluation. |
| Support Vector Machine | To be completed after training and evaluation. |

### Overall winner

**TBD** — select the model after comparing the evaluation metrics and
considering the assignment's performance goals.

## Project structure

.
├── app.py
├── adult.csv
├── test_data.csv
├── requirements.txt
├── README.md
├── model/
│   └── .gitkeep
└── src/
    ├── evaluation.py
    ├── preprocessing.py
    └── train.py

Trained model pipelines will be saved under `model/` after training. The
preprocessing pipeline should be saved with each classifier so inference uses
the same transformations as training.

## Run locally

1. Install the dependencies listed in `requirements.txt`.
2. Implement the training and Streamlit app code.
3. Run the app with `streamlit run app.py`.

## Deploy on Streamlit Community Cloud

Create a new app in Streamlit Community Cloud, select this GitHub repository,
choose the branch containing the app, and set the main file to `app.py`.
Deploy after the app and its required model files are ready.
