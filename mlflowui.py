import os
import sys
import warnings
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Configure logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Helper: compute metrics
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }

# Helper: save confusion matrix plot
def log_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

# Helper: save feature importance plot
def log_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    fi_path = "feature_importance.png"
    plt.savefig(fi_path)
    plt.close()
    mlflow.log_artifact(fi_path)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Load dataset
    csv_path = os.path.join("Data", "synthetic_recurring_payment_data.csv")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.exception("❌ Unable to load dataset: %s", e)
        sys.exit(1)

    # Preprocessing
    df.drop(columns=["CustomerID", "PaymentDate"], inplace=True)
    df["IsWeekend"] = df["IsWeekend"].astype(int)
    df = pd.get_dummies(df, columns=["PaymentType", "AccountType"], drop_first=True)
    df["PaymentStatus"] = df["PaymentStatus"].map({"Success": 0, "Fail": 1})

    # Split dataset
    X = df.drop("PaymentStatus", axis=1)
    y = df["PaymentStatus"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        metrics = compute_metrics(y_test, y_pred)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log params
        mlflow.log_param("n_estimators", 100)

        # Log classification report
        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        # Log confusion matrix and feature importance
        log_confusion_matrix(y_test, y_pred, class_names=["Success", "Fail"])
        log_feature_importance(model, X.columns)

        # Log model
        #signature = infer_signature(X_test, y_pred)
        tracking_type = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_type != "file":
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="RecurringPayModel"
            )
        else:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model"
            )
