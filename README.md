## Machine Learning Assignment 2 for 2026AJ05092 

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


├── app.py                          # Streamlit prediction interface
├── requirements.txt                # Python dependencies
├── setup.py                        # Package configuration
├── notebook/
│   ├── adult_income.ipynb          # Exploratory/training notebook
│   ├── test_data.csv               # Held-out test data
│   ├── data/
│   │   └── adult.csv               # Adult Income source dataset
│   └── model/                      # Saved scikit-learn model pipelines
│       ├── decision_tree.joblib
│       ├── knn.joblib
│       ├── logistic_regression.joblib
│       ├── naive_bayes.joblib
│       └── support_vector_machine.joblib
├── src/
│   ├── __init__.py
│   ├── exception.py                # Shared exception handling
│   ├── logger.py                   # Application logging configuration
│   ├── components
│   │
│   └── pipeline/
│       ├── predict_pipeline.py     # Load artifacts and make predictions
│       └── train_pipeline.py       # Run the training workflow
└── tests/
    └── test_predict_pipeline.py    # Prediction pipeline tests

The Streamlit app loads the five saved pipelines from `notebook/model/`. Each
pipeline includes its fitted preprocessing, so predictions use the same feature
transformations as training. The evaluation section compares all five saved
models on `notebook/test_data.csv` by default, or on an uploaded test CSV (up to
10 MB). The CSV must include the 14 input features and the `income` target.
Evaluation displays accuracy, AUC, precision, recall, F1, MCC, a classification
report, and a confusion matrix for the selected model. Uploaded data is used
only for evaluation; the app never retrains models. `logs/` is used at runtime
for log files; it is not part of the source-code package.
## Run locally

From the project root, create and activate a virtual environment, then install
the dependencies:

powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt


Start the app:

powershell
python -m streamlit run app.py


Streamlit prints a local URL (usually `http://localhost:8501`). Logs are written
to the console and `logs/app.log`. Set `APP_LOG_DIR` to change the log directory.

Run the prediction pipeline tests with:

powershell
python -m pytest


## Deploy on Streamlit Community Cloud

1. Push the app, `requirements.txt`, and all five files in `notebook/model/` to
   the GitHub branch you want to deploy. The saved model files must remain in
   that folder.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app
   from this repository and branch.
3. Set the app's main file path to `app.py`, then deploy.
4. If choosing a Python runtime manually, select Python 3.11 or newer. The
   dependency file pins scikit-learn to the model artifact version.
5. Open the Link as i have deployed on cloud as well 
  https://mlassignment2-nishnat2026aj05092.streamlit.app/



The app does not require secrets, external services, or a separately running
training job. Community Cloud logs are available in the app's management page;
its filesystem is ephemeral, so `logs/app.log` is intended for local debugging.
