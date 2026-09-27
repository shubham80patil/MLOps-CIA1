"""
src/model_training.py
Model training and evaluation with MLflow experiment tracking.
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import xgboost as xgb
import joblib
import os


def get_models() -> dict:
    """Return a dictionary of models to train."""
    models = {
        "Logistic_Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42,
            class_weight="balanced", n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric="logloss",
            scale_pos_weight=5, use_label_encoder=False
        ),
        "Gradient_Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42
        ),
    }
    return models


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    return metrics


def train_and_log_models(models: dict, X_train, X_test, y_train, y_test,
                          experiment_name: str = "Online_Shoppers_Intention",
                          model_save_dir: str = "models") -> dict:
    """
    Train multiple models and log everything to MLflow.

    Returns:
        results (dict): model_name -> metrics dictionary
    """
    os.makedirs(model_save_dir, exist_ok=True)
    mlflow.set_experiment(experiment_name)

    results = {}

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"\n{'='*60}")
            print(f"Training: {name}")
            print(f"{'='*60}")

            # Log model parameters
            params = model.get_params()
            for param_name, param_value in params.items():
                try:
                    mlflow.log_param(param_name, param_value)
                except Exception:
                    pass  # Skip non-serializable params

            # Train
            model.fit(X_train, y_train)

            # Evaluate on train set
            train_metrics = evaluate_model(model, X_train, y_train)
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)

            # Evaluate on test set
            test_metrics = evaluate_model(model, X_test, y_test)
            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)

            # Log model artifact
            mlflow.sklearn.log_model(model, artifact_path=name)

            # Save model locally
            model_path = os.path.join(model_save_dir, f"{name}.joblib")
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)

            # Print results
            print(f"\nTrain Accuracy: {train_metrics['accuracy']:.4f}")
            print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
            print(f"Precision:      {test_metrics['precision']:.4f}")
            print(f"Recall:         {test_metrics['recall']:.4f}")
            print(f"F1-Score:       {test_metrics['f1_score']:.4f}")
            if "roc_auc" in test_metrics:
                print(f"ROC-AUC:        {test_metrics['roc_auc']:.4f}")

            results[name] = {
                "train": train_metrics,
                "test": test_metrics,
                "model": model,
            }

            # Log tags
            mlflow.set_tag("model_type", name)
            mlflow.set_tag("dataset", "Online_Shoppers_Intention")

    return results


def get_best_model(results: dict, metric: str = "f1_score") -> tuple:
    """Find the best model based on test metric."""
    best_name = max(results, key=lambda k: results[k]["test"][metric])
    best_model = results[best_name]["model"]
    best_score = results[best_name]["test"][metric]
    print(f"\nBest model by {metric}: {best_name} ({best_score:.4f})")
    return best_name, best_model, best_score
