# 🚀 MLOps CIA1 — Online Shoppers Purchasing Intention

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-3.x-orange.svg)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-3.x-purple.svg)](https://dvc.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

End-to-end MLOps pipeline for predicting online shoppers' purchasing intention using the **UCI Online Shoppers Purchasing Intention Dataset**. This project demonstrates the complete ML lifecycle with industry-standard MLOps tools.

## 🎯 Assignment Tasks

| Task | Description | Tools |
|------|-------------|-------|
| **Task 1** | MLOps Lifecycle Analysis | Diagram + Code |
| **Task 2** | Experiment Tracking | MLflow |
| **Task 3** | Dataset Versioning | DVC + Git |
| **Task 4** | Workflow Automation | Kubeflow (Simulated) |
| **Task 5A** | Feature Store | Feast (Conceptual) |
| **Task 5B** | Cloud Platform Comparison | Vertex AI vs SageMaker |
| **Bonus** | Model Monitoring | Custom Module |
| **Bonus** | Cloud Deployment Guide | Google Vertex AI |

## 📁 Repository Structure

```
MLOps-CIA1/
├── MLOps_CIA1.ipynb                    # 📓 Main notebook (all tasks)
├── Dataset/
│   ├── online_shoppers_intention.csv   # Original dataset (DVC tracked)
│   └── online_shoppers_intention_v2.csv # Modified dataset (DVC tracked)
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py           # Data loading & preprocessing
│   ├── model_training.py               # Model training with MLflow
│   └── monitoring.py                   # Production monitoring simulation
├── pipeline/
│   └── kubeflow_pipeline.py            # Simulated Kubeflow pipeline
├── feast_feature_store/
│   └── feature_definitions.py          # Feast feature store definitions
├── models/                             # Saved models & visualizations
├── .dvc/                               # DVC configuration
├── .gitignore
├── requirements.txt
└── README.md
```

## 🛠️ Setup & Installation

### Local Setup
```bash
# Clone the repository
git clone https://github.com/shubham80patil/MLOps-CIA1.git
cd MLOps-CIA1

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook MLOps_CIA1.ipynb
```

### Google Colab
The notebook auto-detects Colab and installs dependencies automatically. Just open and run!

## 🔬 Models Trained

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Baseline linear classifier with balanced class weights |
| **Random Forest** | Ensemble of 200 decision trees |
| **XGBoost** | Gradient boosted trees with scale_pos_weight for imbalance |
| **Gradient Boosting** | Sklearn gradient boosting classifier |

All models are tracked via **MLflow** with parameters, metrics, and artifacts.

## 📊 MLflow Experiment Tracking

After running the notebook, view experiments in the MLflow UI:

```bash
cd MLOps-CIA1
mlflow ui --port 5000
```

Then open [http://localhost:5000](http://localhost:5000) to compare models.

**Tracked Metrics:**
- Train/Test Accuracy
- Precision, Recall, F1-Score
- ROC-AUC

## 📁 DVC Data Versioning

```bash
# Version 1: Original dataset
git checkout v1.0-original-dataset

# Version 2: Modified dataset (with feature engineering)
git checkout v2.0-modified-dataset

# Compare versions
dvc diff v1.0-original-dataset v2.0-modified-dataset
```

## ⚙️ Kubeflow Pipeline

Simulated 6-step pipeline:

```
Data Collection → Data Validation → Preprocessing → Training → Evaluation → Deployment
```

## ☁️ Cloud Deployment

Deployment guide for **Google Vertex AI** is included in the notebook with:
- Model upload instructions
- Endpoint creation
- Prediction API
- Monitoring setup

## 📊 Dataset

**UCI Online Shoppers Purchasing Intention Dataset**
- **Samples:** 12,330 browsing sessions
- **Features:** 17 (10 numerical, 7 categorical)
- **Target:** Revenue (True/False) — binary classification
- **Class Imbalance:** ~84.5% No Purchase, ~15.5% Purchase

## 📝 License

This project is created for academic purposes (CIA1 assignment).

---

**Author:** Shubham Patil  
**Course:** MLOps — Trimester 5  
**Institution:** Christ University
