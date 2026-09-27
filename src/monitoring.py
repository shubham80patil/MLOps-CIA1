"""
src/monitoring.py
Model monitoring - simulates production monitoring of a deployed ML model.
Tracks prediction behaviour, performance drift, data drift, and errors.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Prediction Behaviour Monitoring
# ---------------------------------------------------------------------------
def monitor_predictions(model, X_batch: pd.DataFrame) -> dict:
    """
    Monitor prediction distribution for a new batch of data.
    Returns summary stats about predictions.
    """
    y_pred = model.predict(X_batch)
    y_proba = model.predict_proba(X_batch)[:, 1]

    report = {
        "total_predictions": len(y_pred),
        "positive_predictions": int(y_pred.sum()),
        "negative_predictions": int((y_pred == 0).sum()),
        "positive_rate": float(y_pred.mean()),
        "avg_confidence": float(y_proba.mean()),
        "min_confidence": float(y_proba.min()),
        "max_confidence": float(y_proba.max()),
        "std_confidence": float(y_proba.std()),
        "low_confidence_count": int((y_proba > 0.4) & (y_proba < 0.6)).sum(),
    }
    return report


# ---------------------------------------------------------------------------
# 2. Model Performance Monitoring
# ---------------------------------------------------------------------------
def monitor_performance(model, X_batch, y_true, baseline_metrics: dict) -> dict:
    """
    Compare current-batch performance against the baseline (training) metrics.
    Flags drift if performance degrades beyond a threshold.
    """
    y_pred = model.predict(X_batch)

    current = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    drift_report = {}
    THRESHOLD = 0.05  # 5 % drop triggers alert

    for metric_name in current:
        baseline_val = baseline_metrics.get(metric_name, current[metric_name])
        delta = current[metric_name] - baseline_val
        drifted = delta < -THRESHOLD
        drift_report[metric_name] = {
            "baseline": round(baseline_val, 4),
            "current": round(current[metric_name], 4),
            "delta": round(delta, 4),
            "alert": "⚠️ DRIFT DETECTED" if drifted else "✅ OK",
        }

    return drift_report


# ---------------------------------------------------------------------------
# 3. Data Drift Monitoring (simple statistical approach)
# ---------------------------------------------------------------------------
def monitor_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                       numeric_cols: list = None) -> dict:
    """
    Compare distributions of numeric features between reference and current data.
    Uses mean/std shift as a simple proxy for drift.
    """
    if numeric_cols is None:
        numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()

    drift_report = {}
    for col in numeric_cols:
        ref_mean = reference_df[col].mean()
        cur_mean = current_df[col].mean()
        ref_std = reference_df[col].std()

        # Normalised mean shift
        shift = abs(cur_mean - ref_mean) / (ref_std + 1e-8)
        drifted = shift > 1.0  # More than 1 std deviation of shift

        drift_report[col] = {
            "ref_mean": round(ref_mean, 4),
            "cur_mean": round(cur_mean, 4),
            "normalized_shift": round(shift, 4),
            "alert": "⚠️ DATA DRIFT" if drifted else "✅ OK",
        }

    return drift_report


# ---------------------------------------------------------------------------
# 4. Error Monitoring
# ---------------------------------------------------------------------------
def monitor_errors(model, X_batch: pd.DataFrame) -> dict:
    """
    Attempt predictions and catch / log errors for monitoring purposes.
    """
    errors = []
    successful = 0

    for i in range(len(X_batch)):
        try:
            row = X_batch.iloc[[i]]
            model.predict(row)
            successful += 1
        except Exception as e:
            errors.append({"index": i, "error": str(e)})

    report = {
        "total_samples": len(X_batch),
        "successful_predictions": successful,
        "failed_predictions": len(errors),
        "error_rate": len(errors) / len(X_batch) if len(X_batch) > 0 else 0,
        "errors": errors[:10],  # Log first 10 errors only
    }
    return report


# ---------------------------------------------------------------------------
# 5. Full Monitoring Dashboard
# ---------------------------------------------------------------------------
def run_full_monitoring(model, X_ref, X_batch, y_batch_true,
                        baseline_metrics: dict) -> dict:
    """Run all monitoring checks and return a consolidated report."""
    print("=" * 70)
    print("           MODEL MONITORING DASHBOARD")
    print("=" * 70)

    # Prediction behaviour
    pred_report = monitor_predictions(model, X_batch)
    print("\n📊 Prediction Behaviour")
    print(f"   Total predictions : {pred_report['total_predictions']}")
    print(f"   Positive rate     : {pred_report['positive_rate']:.2%}")
    print(f"   Avg confidence    : {pred_report['avg_confidence']:.4f}")
    print(f"   Low-confidence    : {pred_report['low_confidence_count']}")

    # Performance
    perf_report = monitor_performance(model, X_batch, y_batch_true, baseline_metrics)
    print("\n📈 Performance Monitoring")
    for metric, info in perf_report.items():
        print(f"   {metric:12s}  baseline={info['baseline']:.4f}  "
              f"current={info['current']:.4f}  delta={info['delta']:+.4f}  "
              f"{info['alert']}")

    # Data drift
    numeric_cols = X_ref.select_dtypes(include=[np.number]).columns.tolist()
    data_report = monitor_data_drift(X_ref, X_batch, numeric_cols)
    drift_features = [c for c, v in data_report.items() if "DRIFT" in v["alert"]]
    print(f"\n📉 Data Drift: {len(drift_features)} feature(s) drifted out of {len(numeric_cols)}")
    for col in drift_features:
        info = data_report[col]
        print(f"   {col}: shift={info['normalized_shift']:.4f}")

    # Errors
    error_report = monitor_errors(model, X_batch)
    print(f"\n🔴 Error Monitoring")
    print(f"   Success rate: {error_report['successful_predictions']}/{error_report['total_samples']}")
    print(f"   Error rate  : {error_report['error_rate']:.2%}")

    print("\n" + "=" * 70)

    return {
        "predictions": pred_report,
        "performance": perf_report,
        "data_drift": data_report,
        "errors": error_report,
    }
