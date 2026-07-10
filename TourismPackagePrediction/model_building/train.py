# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# for model serialization
import joblib

# for creating a folder
import os

# Hugging Face
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

import mlflow

# MLflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-prediction")

# Hugging Face

token = os.getenv("HF_MLOps")
api = HfApi(token=os.getenv("HF_MLOps"))

# Download Dataset Files

repo_id = "mesan21ster/TourismPackagePrediction"

Xtrain = pd.read_csv(
    hf_hub_download(
        repo_id=repo_id,
        filename="Xtrain.csv",
        repo_type="dataset",
        token=token
    )
)

Xtest = pd.read_csv(
    hf_hub_download(
        repo_id=repo_id,
        filename="Xtest.csv",
        repo_type="dataset",
        token=token
    )
)

ytrain = pd.read_csv(
    hf_hub_download(
        repo_id=repo_id,
        filename="ytrain.csv",
        repo_type="dataset",
        token=token
    )
).squeeze()

ytest = pd.read_csv(
    hf_hub_download(
        repo_id=repo_id,
        filename="ytest.csv",
        repo_type="dataset",
        token=token
    )
).squeeze()

# Feature Lists

numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome"
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"
]

# Class Weight

class_weight = (
    ytrain.value_counts()[0]
    / ytrain.value_counts()[1]
)

# Preprocessor

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# XGBoost

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42
)

# Hyperparameter Grid

param_grid = {
    'xgbclassifier__n_estimators': [50, 75],
    'xgbclassifier__max_depth': [2, 3],
    'xgbclassifier__colsample_bytree': [0.4, 0.5],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5],
    'xgbclassifier__learning_rate': [0.01, 0.05],
    'xgbclassifier__reg_lambda': [0.4, 0.5],
}

# Pipeline

model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)

# MLflow Training

with mlflow.start_run():

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(Xtrain, ytrain)

    # Log Grid Search Results

    results = grid_search.cv_results_

    for i in range(len(results["params"])):

        with mlflow.start_run(nested=True):

            mlflow.log_params(results["params"][i])

            mlflow.log_metric(
                "mean_test_score",
                results["mean_test_score"][i]
            )

            mlflow.log_metric(
                "std_test_score",
                results["std_test_score"][i]
            )

    # Best Model

    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # Prediction

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Evaluation

    train_report = classification_report(
        ytrain,
        y_pred_train,
        output_dict=True
    )

    test_report = classification_report(
        ytest,
        y_pred_test,
        output_dict=True
    )

    mlflow.log_metrics({

        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],

        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"]

    })

    # Save Model

    model_path = "best_tourism_package_model_v1.joblib"

    joblib.dump(best_model, model_path)

    mlflow.log_artifact(
        model_path,
        artifact_path="model"
    )

    print(f"Model saved as artifact: {model_path}")

    # Upload to Hugging Face Model Hub
  
    model_repo = "mesan21ster/tourism_package_prediction_model"

    try:
        api.repo_info(
            repo_id=model_repo,
            repo_type="model"
        )

        print(f"Model repository '{model_repo}' already exists.")

    except RepositoryNotFoundError:

        print(f"Creating model repository '{model_repo}'...")

        api.create_repo(
            repo_id=model_repo,
            repo_type="model",
            token=token,
            private=False,
            exist_ok=True
        )

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=model_repo,
        repo_type="model"
    )

    print("Model uploaded successfully.")
