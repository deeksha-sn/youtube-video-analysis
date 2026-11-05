"""
Decision Tree Classification Module
====================================
Course: Data Mining & Data Warehousing (BCI24505D)
Topic 42: YouTube Video Analysis

Implements Algorithm 2: Supervised Decision Tree Classification to predict
viewer engagement levels using strictly pre-publication video metadata.
Guarantees ZERO DATA LEAKAGE by excluding post-upload engagement metrics
(views, likes, comments) from predictors.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from typing import Dict, Any, Tuple

from src.preprocessing import CLEANED_CSV_PATH

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "figures")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "models")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "reports")
CLASSIFICATION_REPORT_PATH = os.path.join(REPORTS_DIR, "classification_report.json")


def run_classification(csv_path: str = CLEANED_CSV_PATH) -> Tuple[DecisionTreeClassifier, Dict[str, Any]]:
    """Train, evaluate, and interpret a Decision Tree Classifier on video metadata."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} records for Decision Tree classification.")
    
    # -------------------------------------------------------------------------
    # Target Variable: high_engagement (1 = High Engagement, 0 = Low/Normal)
    # -------------------------------------------------------------------------
    if "high_engagement" not in df.columns:
        median_eng = df["engagement_rate"].median()
        df["high_engagement"] = (df["engagement_rate"] >= median_eng).astype(int)
    
    y = df["high_engagement"]
    
    # -------------------------------------------------------------------------
    # Feature Selection (ZERO DATA LEAKAGE)
    # Strictly exclude views, likes, dislikes, comments, and engagement ratios!
    # -------------------------------------------------------------------------
    feature_cols = [
        "publish_hour", "publish_weekday",
        "title_length", "tag_count", "desc_length"
    ]
    
    # One-hot encode the categorical 'category_name'
    cat_dummies = pd.get_dummies(df["category_name"], prefix="cat", drop_first=True, dtype=int)
    X = pd.concat([df[feature_cols], cat_dummies], axis=1)
    
    print(f"[INFO] Total predictor features: {X.shape[1]}")
    print(f"[INFO] Class balance: {y.value_counts().to_dict()}")
    
    # Train / Test Split (80% train, 20% test with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Initialize and train Decision Tree with controlled depth for explainability
    dt_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        min_samples_leaf=25,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    
    # Save trained model
    model_path = os.path.join(MODELS_DIR, "decision_tree_model.joblib")
    joblib.dump(dt_model, model_path)
    
    # Predictions
    y_pred = dt_model.predict(X_test)
    
    # Evaluation Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n[METRICS] Decision Tree Accuracy:  {acc * 100:.2f}%")
    print(f"[METRICS] Decision Tree Precision: {prec * 100:.2f}%")
    print(f"[METRICS] Decision Tree Recall:    {rec * 100:.2f}%")
    print(f"[METRICS] Decision Tree F1-Score:  {f1 * 100:.2f}%")
    
    # -------------------------------------------------------------------------
    # Visualization 1: Confusion Matrix
    # -------------------------------------------------------------------------
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted Normal", "Predicted High"],
        yticklabels=["Actual Normal", "Actual High"]
    )
    plt.title("Decision Tree Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "15_decision_tree_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    
    # -------------------------------------------------------------------------
    # Visualization 2: Decision Tree Structure
    # -------------------------------------------------------------------------
    plt.figure(figsize=(20, 10))
    plot_tree(
        dt_model,
        feature_names=list(X.columns),
        class_names=["Normal", "High Eng"],
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=3
    )
    plt.title("Trained Decision Tree Visualization (Depth 3 Preview)")
    plt.tight_layout()
    tree_path = os.path.join(FIGURES_DIR, "16_decision_tree_visualization.png")
    plt.savefig(tree_path)
    plt.close()
    
    # -------------------------------------------------------------------------
    # Visualization 3: Feature Importances
    # -------------------------------------------------------------------------
    importances = pd.Series(dt_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_importances = importances.head(10).reset_index()
    top_importances.columns = ["feature", "importance"]
    
    plt.figure(figsize=(9, 5))
    bar_fi = sns.barplot(
        data=top_importances,
        y="feature",
        x="importance",
        palette="viridis",
        hue="feature",
        legend=False
    )
    plt.title("Top 10 Feature Importances (Gini Impurity Reduction)")
    plt.xlabel("Relative Importance")
    plt.ylabel("Metadata Feature")
    for p in bar_fi.patches:
        width = p.get_width()
        bar_fi.annotate(f"{width:.3f}",
                        (width, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', xytext=(5, 0), textcoords='offset points', fontsize=9)
    plt.tight_layout()
    fi_path = os.path.join(FIGURES_DIR, "17_decision_tree_feature_importance.png")
    plt.savefig(fi_path)
    plt.close()
    
    report = {
        "algorithm": "Decision Tree Classifier",
        "parameters": {
            "criterion": "gini",
            "max_depth": 4,
            "min_samples_leaf": 25,
            "random_state": 42
        },
        "evaluation_metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": {
                "true_negative": int(cm[0, 0]),
                "false_positive": int(cm[0, 1]),
                "false_negative": int(cm[1, 0]),
                "true_positive": int(cm[1, 1])
            }
        },
        "feature_importances": importances.to_dict(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"[SUCCESS] Classification report saved to: {CLASSIFICATION_REPORT_PATH}")
    
    return dt_model, report


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 5: DECISION TREE CLASSIFICATION")
    print("=" * 60)
    model, clf_rep = run_classification()
