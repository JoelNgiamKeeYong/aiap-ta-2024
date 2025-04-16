import os
import time
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tabulate import tabulate
import matplotlib.pyplot as plt


def train_and_evaluate_models(
        X_train, X_test, y_train, y_test,
        n_jobs, cv_folds, model_configuration,
        scoring_metric, random_state, feature_names=None
):
    """
    Train and evaluate multiple machine learning models for no-show prediction.

    Args:
        X_train (pandas.DataFrame or numpy.ndarray): Training feature matrix.
        X_test (pandas.DataFrame or numpy.ndarray): Testing feature matrix.
        y_train (pandas.Series or numpy.ndarray): Training target variable.
        y_test (pandas.Series or numpy.ndarray): Testing target variable.
        n_jobs (int): Number of parallel jobs for GridSearchCV.
        cv_folds (int): Number of cross-validation folds.
        model_configuration (dict): Dictionary containing model configurations.
        scoring_metric (str): Scoring metric for GridSearchCV.
        feature_names (list): List of feature names (optional, for sparse matrices).

    Returns:
        dict: Dictionary containing trained models and their evaluation metrics.
    """
    # Define the models and their respective hyperparameter grids
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=random_state),
            "params": model_configuration['Logistic Regression']['params']
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric='logloss', random_state=random_state),
            "params": model_configuration['XGBoost']['params']
        },
        "LightGBM": {  
            "model": LGBMClassifier(verbose=-1, force_row_wise=True, random_state=random_state),
            "params": model_configuration['LightGBM']['params'] 
        }
    }

    # Ensure the output directory exists
    os.makedirs("output", exist_ok=True)

    # Lists to store results
    results = []  # Evaluation metrics for each model
    feature_importance_results = []  # Feature importance results
    roc_data = []  # ROC curve data for all models
    pr_data = []  # Precision-Recall curve data for all models

    # Loop through each model and perform hyperparameter tuning
    for model_name, model_info in models.items():
        print(f"\n🛠️  Training {model_name}...")
        start_time = time.time()

        best_model = perform_hyperparameter_tuning(
            model_info["model"], model_info["params"],
            X_train, y_train, scoring_metric, cv_folds, n_jobs
        )

        # Evaluate the model step-by-step
        train_metrics, test_metrics = compute_metrics(best_model, X_train, X_test, y_train, y_test)
        formatted_metrics = format_metrics(model_name, test_metrics, train_metrics)

        roc_auc = compute_roc_auc(best_model, X_test, y_test, model_name)
        if roc_auc:
            roc_data.append(roc_auc)
            save_roc_curve(roc_auc)

        pr_auc = compute_precision_recall(best_model, X_test, y_test, model_name)
        if pr_auc:
            pr_data.append(pr_auc)
            save_pr_curve(pr_auc)

        extract_and_save_feature_importance(best_model, model_name, feature_names, feature_importance_results)

        # Save consolidated metrics for the current model
        results.append(formatted_metrics)
        save_evaluation_metrics_summary(results)

        end_time = time.time() 
        elapsed_time = end_time - start_time
        print(f"✅ Completed in {elapsed_time:.2f} seconds. {model_name} trained successfully!")

    return results


def perform_hyperparameter_tuning(model, params, X_train, y_train, scoring_metric, cv_folds, n_jobs):
    """
    Perform hyperparameter tuning using GridSearchCV.

    Args:
        model: The machine learning model.
        params (dict): Hyperparameter grid for the model.
        X_train (pandas.DataFrame or numpy.ndarray): Training feature matrix.
        y_train (pandas.Series or numpy.ndarray): Training target variable.
        scoring_metric (str): Scoring metric for GridSearchCV.
        cv_folds (int): Number of cross-validation folds.
        n_jobs (int): Number of parallel jobs.

    Returns:
        object: Best model after hyperparameter tuning.
    """
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=params, scoring=scoring_metric, cv=cv_folds, n_jobs=n_jobs)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def compute_metrics(model, X_train, X_test, y_train, y_test):
    """
    Compute evaluation metrics for training and test data.

    Args:
        model: The trained machine learning model.
        X_train (pandas.DataFrame or numpy.ndarray): Training feature matrix.
        X_test (pandas.DataFrame or numpy.ndarray): Testing feature matrix.
        y_train (pandas.Series or numpy.ndarray): Training target variable.
        y_test (pandas.Series or numpy.ndarray): Testing target variable.

    Returns:
        tuple: Test metrics and training metrics as dictionaries.
    """
    def calculate_metrics(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred),
        }

    train_metrics = calculate_metrics(y_train, model.predict(X_train))
    test_metrics = calculate_metrics(y_test, model.predict(X_test))

    return train_metrics, test_metrics


def compute_roc_auc(model, X_test, y_test, model_name):
    """
    Compute ROC curve data for the current model.

    Args:
        model: The trained machine learning model.
        X_test (pandas.DataFrame or numpy.ndarray): Testing feature matrix.
        y_test (pandas.Series or numpy.ndarray): Testing target variable.
        model_name (str): Name of the model.

    Returns:
        dict: ROC curve data (FPR, TPR, AUC).
    """
    try:
        y_test_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_test_probs)
        roc_auc = auc(fpr, tpr)
        return {"Model": model_name, "FPR": fpr, "TPR": tpr, "AUC": roc_auc}
    except AttributeError:
        print(f" └── ⚠️ Model {model_name} does not support `predict_proba`. Skipping ROC curve generation.")
        return None


def compute_precision_recall(model, X_test, y_test, model_name):
    """
    Compute Precision-Recall curve data for the current model.

    Args:
        model: The trained machine learning model.
        X_test (pandas.DataFrame or numpy.ndarray): Testing feature matrix.
        y_test (pandas.Series or numpy.ndarray): Testing target variable.
        model_name (str): Name of the model.

    Returns:
        dict: Precision-Recall curve data (Precision, Recall, AUC-PR).
    """
    try:
        y_test_probs = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_test_probs)
        auc_pr = auc(recall, precision)
        return {"Model": model_name, "Precision": precision, "Recall": recall, "AUC-PR": auc_pr}
    except AttributeError:
        print(f" └── ⚠️ Model {model_name} does not support `predict_proba`. Skipping Precision-Recall curve generation.")
        return None


def extract_and_save_feature_importance(model, model_name, feature_names, feature_importance_results):
    """
    Extract and save feature importance scores (if supported by the model).

    Args:
        model: The trained machine learning model.
        model_name (str): Name of the model.
        feature_names (list): List of feature names.
        feature_importance_results (list): List to store feature importance results.
    """
    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        feature_importance_results.append({"Model": model_name, "Feature Importances": feature_importances.to_dict()})
        save_feature_importance(model_name, feature_importances)


def save_feature_importance(model_name, feature_importances):
    """
    Save feature importance scores for a model to a file.

    Args:
        model_name (str): Name of the model.
        feature_importances (pandas.Series): Feature importance scores.
    """
    os.makedirs("output/feature_importance", exist_ok=True)
    file_path = f"output/feature_importance/{model_name.replace(' ', '_').lower()}_feature_importances.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"📊 Feature Importance Scores for {model_name}:\n")
        f.write(feature_importances.to_string())
    print(f"💾 Saved feature importance for {model_name} to {file_path}")


def save_roc_curve(roc_auc):
    """
    Save an individual ROC curve as a .png file.

    Args:
        roc_auc (dict): ROC curve data (FPR, TPR, AUC).
    """
    model_name = roc_auc["Model"]
    file_path = f"output/{model_name.replace(' ', '_').lower()}_roc_curve.png"
    plt.figure(figsize=(8, 6))
    plt.plot(roc_auc["FPR"], roc_auc["TPR"], label=f"AUC = {roc_auc['AUC']:.2f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier", linewidth=1.5)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curve - {model_name}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(color="lightgray", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved ROC curve for {model_name} to {file_path}")


def save_pr_curve(pr_auc):
    """
    Save an individual Precision-Recall curve as a .png file.

    Args:
        pr_auc (dict): Precision-Recall curve data (Precision, Recall, AUC-PR).
    """
    model_name = pr_auc["Model"]
    file_path = f"output/{model_name.replace(' ', '_').lower()}_pr_curve.png"
    plt.figure(figsize=(8, 6))
    plt.plot(pr_auc["Recall"], pr_auc["Precision"], label=f"AUC-PR = {pr_auc['AUC-PR']:.2f}", linewidth=2)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curve - {model_name}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(color="lightgray", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved Precision-Recall curve for {model_name} to {file_path}")


def save_evaluation_metrics_summary(results):
    """
    Save consolidated evaluation metrics summary to a .txt file.

    Args:
        results (list): List of evaluation metrics for each model.
    """
    metrics_file_path = "output/evaluation_metrics_summary.txt"
    with open(metrics_file_path, "w", encoding="utf-8") as f:
        f.write("📋 Consolidated Evaluation Metrics:\n")
        metrics_table = tabulate(pd.DataFrame(results), headers="keys", tablefmt="grid")
        f.write(metrics_table + "\n\n")
    print(f"💾 Saved consolidated evaluation metrics to {metrics_file_path}")


def format_metrics(model_name, test_metrics, train_metrics):
    """
    Format evaluation metrics into a dictionary.

    Args:
        model_name (str): Name of the model.
        test_metrics (dict): Test metrics.
        train_metrics (dict): Training metrics.

    Returns:
        dict: Formatted metrics.
    """
    return {
        "Model": model_name,
        "Test Accuracy": test_metrics["Accuracy"],
        "Train Accuracy": train_metrics["Accuracy"],
        "Test Precision": test_metrics["Precision"],
        "Train Precision": train_metrics["Precision"],
        "Test Recall": test_metrics["Recall"],
        "Train Recall": train_metrics["Recall"],
        "Test F1-Score": test_metrics["F1-Score"],
        "Train F1-Score": train_metrics["F1-Score"],
    }