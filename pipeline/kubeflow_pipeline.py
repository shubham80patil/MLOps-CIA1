"""
pipeline/kubeflow_pipeline.py
Simulated Kubeflow Pipeline for the Online Shoppers Intention ML workflow.

This script demonstrates pipeline concepts (DAG of components) without
requiring a live Kubeflow cluster. Each step is implemented as a Python
function that mirrors what a Kubeflow ContainerOp would do.
"""

import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime


# ============================================================================
# Pipeline Components (simulated Kubeflow ContainerOps)
# ============================================================================

def step_1_data_collection(data_path: str) -> dict:
    """
    STEP 1: Data Collection
    - Read raw data from CSV
    - Validate file existence and basic integrity
    """
    print("=" * 60)
    print("STEP 1: DATA COLLECTION")
    print("=" * 60)

    start = time.time()

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    elapsed = time.time() - start

    result = {
        "status": "SUCCESS",
        "rows": df.shape[0],
        "columns": df.shape[1],
        "file_size_kb": round(os.path.getsize(data_path) / 1024, 2),
        "duration_sec": round(elapsed, 3),
        "timestamp": datetime.now().isoformat(),
    }
    print(f"  → Loaded {result['rows']} rows, {result['columns']} columns")
    print(f"  → File size: {result['file_size_kb']} KB")
    return result


def step_2_data_validation(data_path: str) -> dict:
    """
    STEP 2: Data Validation
    - Check for missing values
    - Validate column types
    - Check class balance
    """
    print("\n" + "=" * 60)
    print("STEP 2: DATA VALIDATION")
    print("=" * 60)

    df = pd.read_csv(data_path)

    missing = df.isnull().sum()
    total_missing = int(missing.sum())

    expected_columns = [
        "Administrative", "Administrative_Duration", "Informational",
        "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues", "SpecialDay",
        "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
        "VisitorType", "Weekend", "Revenue",
    ]
    schema_valid = all(col in df.columns for col in expected_columns)

    class_dist = df["Revenue"].value_counts().to_dict()
    imbalance_ratio = round(class_dist.get(False, 0) / max(class_dist.get(True, 1), 1), 2)

    result = {
        "status": "SUCCESS" if schema_valid else "FAILED",
        "schema_valid": schema_valid,
        "total_missing_values": total_missing,
        "missing_by_column": {k: int(v) for k, v in missing[missing > 0].items()},
        "class_distribution": {str(k): int(v) for k, v in class_dist.items()},
        "imbalance_ratio": imbalance_ratio,
        "duplicate_rows": int(df.duplicated().sum()),
        "timestamp": datetime.now().isoformat(),
    }

    print(f"  → Schema valid: {schema_valid}")
    print(f"  → Total missing: {total_missing}")
    print(f"  → Duplicates: {result['duplicate_rows']}")
    print(f"  → Imbalance ratio: {imbalance_ratio}")
    return result


def step_3_data_preprocessing(data_path: str) -> dict:
    """
    STEP 3: Data Preprocessing
    - Encode categoricals
    - Handle missing values
    - Feature scaling
    """
    print("\n" + "=" * 60)
    print("STEP 3: DATA PREPROCESSING")
    print("=" * 60)

    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(data_path)
    initial_rows = len(df)

    # Drop missing
    df.dropna(inplace=True)

    # Encode
    month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6,
                 "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    df["Month"] = df["Month"].map(month_map)
    le = LabelEncoder()
    df["VisitorType"] = le.fit_transform(df["VisitorType"])
    df["Weekend"] = df["Weekend"].astype(int)
    df["Revenue"] = df["Revenue"].astype(int)

    X = df.drop("Revenue", axis=1)
    y = df["Revenue"]

    scaler = StandardScaler()
    X[X.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(
        X.select_dtypes(include=[np.number])
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    result = {
        "status": "SUCCESS",
        "rows_before": initial_rows,
        "rows_after": len(df),
        "rows_dropped": initial_rows - len(df),
        "features": X.shape[1],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "timestamp": datetime.now().isoformat(),
    }

    print(f"  → Dropped {result['rows_dropped']} rows with missing values")
    print(f"  → Features: {result['features']}")
    print(f"  → Train: {result['train_size']} | Test: {result['test_size']}")
    return result


def step_4_model_training(data_path: str) -> dict:
    """
    STEP 4: Model Training (simulated — just shows metrics)
    """
    print("\n" + "=" * 60)
    print("STEP 4: MODEL TRAINING")
    print("=" * 60)

    models_trained = [
        {"name": "Logistic_Regression", "type": "linear"},
        {"name": "Random_Forest", "type": "ensemble"},
        {"name": "XGBoost", "type": "boosting"},
        {"name": "Gradient_Boosting", "type": "boosting"},
    ]

    result = {
        "status": "SUCCESS",
        "models_trained": len(models_trained),
        "model_names": [m["name"] for m in models_trained],
        "tracking_tool": "MLflow",
        "experiment_name": "Online_Shoppers_Intention",
        "timestamp": datetime.now().isoformat(),
    }

    for m in models_trained:
        print(f"  → Trained: {m['name']} ({m['type']})")
    print(f"  → All models logged to MLflow")
    return result


def step_5_model_evaluation(data_path: str) -> dict:
    """
    STEP 5: Model Evaluation
    """
    print("\n" + "=" * 60)
    print("STEP 5: MODEL EVALUATION")
    print("=" * 60)

    result = {
        "status": "SUCCESS",
        "evaluation_metrics": ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
        "best_model_selection_criterion": "f1_score",
        "comparison_tool": "MLflow UI",
        "timestamp": datetime.now().isoformat(),
    }

    print("  → Metrics computed: accuracy, precision, recall, f1, roc-auc")
    print("  → Best model selected by F1-score")
    return result


def step_6_deployment(data_path: str) -> dict:
    """
    STEP 6: Deployment readiness
    """
    print("\n" + "=" * 60)
    print("STEP 6: DEPLOYMENT")
    print("=" * 60)

    result = {
        "status": "SUCCESS",
        "deployment_target": "Google Vertex AI",
        "model_format": "joblib / MLflow Model",
        "serving_type": "REST API Endpoint",
        "monitoring_enabled": True,
        "timestamp": datetime.now().isoformat(),
    }

    print("  → Target: Google Vertex AI")
    print("  → Serving: REST API Endpoint")
    print("  → Monitoring: Enabled")
    return result


# ============================================================================
# Pipeline Runner
# ============================================================================

def run_pipeline(data_path: str) -> dict:
    """
    Run the full simulated Kubeflow pipeline.
    """
    print("\n" + "🔄" * 30)
    print("  KUBEFLOW PIPELINE EXECUTION START")
    print("🔄" * 30 + "\n")

    pipeline_start = time.time()

    results = {}
    steps = [
        ("data_collection", step_1_data_collection),
        ("data_validation", step_2_data_validation),
        ("data_preprocessing", step_3_data_preprocessing),
        ("model_training", step_4_model_training),
        ("model_evaluation", step_5_model_evaluation),
        ("deployment", step_6_deployment),
    ]

    for step_name, step_fn in steps:
        try:
            results[step_name] = step_fn(data_path)
        except Exception as e:
            results[step_name] = {"status": "FAILED", "error": str(e)}
            print(f"  ❌ Step {step_name} FAILED: {e}")
            break

    total_time = round(time.time() - pipeline_start, 3)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — Total time: {total_time}s")
    print(f"{'='*60}")

    results["pipeline_summary"] = {
        "total_steps": len(steps),
        "completed_steps": sum(1 for v in results.values()
                               if isinstance(v, dict) and v.get("status") == "SUCCESS"),
        "total_duration_sec": total_time,
    }

    return results


if __name__ == "__main__":
    run_pipeline("Dataset/online_shoppers_intention.csv")
