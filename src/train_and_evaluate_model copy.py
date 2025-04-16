# src/train_and_evaluate_model.py

import joblib
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb 
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate


def train_and_evaluate_models(
        X_train, X_test, y_train, y_test,
        n_jobs, cv_folds, model_configuration,
        scoring_metric, random_state, feature_names=None
):
    """
    Train and evaluate multiple machine learning models for no-show prediction.

    Args:
        X_train: Training feature matrix
        X_test: Testing feature matrix
        y_train: Training target variable
        y_test: Testing target variable
        n_jobs: Number of parallel jobs for GridSearchCV
        cv_folds: Number of cross-validation folds
        model_configuration: Dictionary containing model configurations
        scoring_metric: Scoring metric for GridSearchCV
        random_state: Random state for reproducibility
        feature_names: List of feature names (optional, for sparse matrices)

    Returns:
        dict: Dictionary containing trained models and their evaluation metrics
    """
    # Define the models and their respective hyperparameter grids
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=random_state),
            "params": model_configuration['Logistic Regression']['params']
        },
        # "Random Forest": {
        #     "model": RandomForestClassifier(random_state=42),
        #     "params": model_configuration['Random Forest']['params']
        # },
        "XGBoost": {
            "model": xgb.XGBClassifier(eval_metric='logloss', random_state=random_state),
            "params": model_configuration['XGBoost']['params']
        },
        "LightGBM": {  
            "model": lgb.LGBMClassifier(verbose=-1, force_row_wise=True, random_state=random_state),
            "params": model_configuration['LightGBM']['params'] 
        }
    }

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Storage for results
    results = []
    feature_importance_results = []
    roc_data = []
    pr_data = [] 
    confusion_matrices = {}

    # Process each model
    for model_name, model_info in models.items():
        print(f"\n🛠️  Training {model_name}...")
        start_time = time.time()

        # Train model with hyperparameter tuning
        best_model = train_model_with_tuning(
            model_info["model"], model_info["params"],
            X_train, y_train, scoring_metric, cv_folds, n_jobs
        )
        
        # Evaluate and store results
        model_results = evaluate_model_comprehensively(
            best_model, model_name, X_train, X_test, y_train, y_test, feature_names
        )
        
        # Store results
        results.append(model_results["metrics"])
        if model_results["feature_importance"] is not None:
            feature_importance_results.append({
                "Model": model_name,
                "Feature Importances": model_results["feature_importance"]
            })
        if model_results["roc_data"] is not None:
            roc_data.append(model_results["roc_data"])
        if model_results["pr_data"] is not None:  # New: Store PR curve data
            pr_data.append(model_results["pr_data"])
        confusion_matrices[model_name] = model_results["confusion_matrix"]
        
        # Save model
        save_model(best_model, model_name)

        # Report timing
        elapsed_time = time.time() - start_time
        print(f"✅ Completed in {elapsed_time:.2f} seconds. {model_name} trained successfully!")

    # Save consolidated results
    save_all_results(results, confusion_matrices, feature_importance_results, roc_data, pr_data)

    return results


def train_model_with_tuning(model, params, X_train, y_train, scoring_metric, cv_folds, n_jobs):
    """
    Perform hyperparameter tuning and return the best model.
    
    Args:
        model: Base model to tune
        params: Parameter grid
        X_train, y_train: Training data
        scoring_metric: Metric for evaluation
        cv_folds: Number of CV folds
        n_jobs: Number of parallel jobs
    
    Returns:
        Best model after tuning
    """
    print(f"   └── Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring=scoring_metric,
        cv=cv_folds,
        n_jobs=n_jobs,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    print(f"   └── Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def evaluate_model_comprehensively(model, model_name, X_train, X_test, y_train, y_test, feature_names=None):
    """
    Comprehensive model evaluation including metrics, feature importance, and visualizations.
    
    Args:
        model: Trained model
        model_name: Name of the model
        X_train, y_train: Training data
        X_test, y_test: Test data
        feature_names: Names of features
    
    Returns:
        Dictionary with all evaluation results
    """
    print(f"   └── Evaluating model...")
    results = {}
    
    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_basic_metrics(y_train, y_train_pred)
    test_metrics = calculate_basic_metrics(y_test, y_test_pred)
    
    # Get probability predictions if possible (for ROC and PR curves)
    roc_data = None
    pr_data = None
    roc_auc = None
    pr_auc = None
    
    try:
        y_test_probs = model.predict_proba(X_test)[:, 1]
        
        # ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_test_probs)
        roc_auc = auc(fpr, tpr)
        roc_data = {"Model": model_name, "FPR": fpr, "TPR": tpr, "AUC": roc_auc}
        
        # Precision-Recall curve data
        precision, recall, _ = precision_recall_curve(y_test, y_test_probs)
        pr_auc = average_precision_score(y_test, y_test_probs)
        pr_data = {
            "Model": model_name,
            "Precision": precision,
            "Recall": recall,
            "AP": pr_auc
        }
    except AttributeError:
        print(f"   └── ⚠️ Model {model_name} does not support `predict_proba`. Skipping curve analysis.")
    
    # Format metrics for output
    formatted_metrics = {
        "Model": model_name,
        "Accuracy": f"{test_metrics['accuracy']:.2f} ({train_metrics['accuracy']:.2f})",
        "Precision": f"{test_metrics['precision']:.2f} ({train_metrics['precision']:.2f})",
        "Recall": f"{test_metrics['recall']:.2f} ({train_metrics['recall']:.2f})",
        "F1-Score": f"{test_metrics['f1']:.2f} ({train_metrics['f1']:.2f})",
        "ROC AUC": f"{roc_auc:.2f}" if roc_auc is not None else "N/A",
        "PR AUC": f"{pr_auc:.2f}" if pr_auc is not None else "N/A"  # Added PR AUC metric
    }
    
    # Extract feature importance
    feature_importance = extract_feature_importance(model, feature_names)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    confusion_table = [
        ["", "Predicted No-Show", "Predicted Show"],
        ["Actual No-Show", tp, fn],
        ["Actual Show", fp, tn]
    ]
    formatted_cm = tabulate(confusion_table, headers="firstrow", tablefmt="grid")
    
    # Compile all results
    results["metrics"] = formatted_metrics
    results["feature_importance"] = feature_importance
    results["roc_data"] = roc_data
    results["pr_data"] = pr_data  # Added PR curve data
    results["confusion_matrix"] = formatted_cm
    
    return results


def calculate_basic_metrics(y_true, y_pred):
    """
    Calculate basic classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary of metrics
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='binary'),
        "recall": recall_score(y_true, y_pred, average='binary'),
        "f1": f1_score(y_true, y_pred, average='binary')
    }


def extract_feature_importance(model, feature_names):
    """
    Extract feature importance from model if available.
    
    Args:
        model: Trained model
        feature_names: Names of features
    
    Returns:
        Dictionary of feature importances or None
    """
    print(f"   └── Extracting feature importance (if applicable)...")
    if feature_names is None:
        print("   └── ⚠️ Feature names not provided, skipping feature importance.")
        return None
        
    if hasattr(model, "coef_"):  # Logistic Regression
        return pd.Series(abs(model.coef_[0]), index=feature_names).sort_values(ascending=False).to_dict()
    elif hasattr(model, "feature_importances_"):  # Tree-based models
        return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).to_dict()
    else:
        return None


def save_model(model, model_name):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model
        model_name: Name of the model
    """
    model_path = f"models/{model_name.replace(' ', '_').lower()}_model.joblib"
    print(f"💾 Saving trained model to {model_path}...")
    joblib.dump(model, model_path)


def save_all_results(results, confusion_matrices, feature_importance_results, roc_data, pr_data):
    """
    Save all evaluation results to files.
    
    Args:
        results: Model metrics
        confusion_matrices: Confusion matrices
        feature_importance_results: Feature importance data
        roc_data: ROC curve data
        pr_data: Precision-Recall curve data
    """
    # 1. Save metrics and confusion matrices
    metrics_file_path = "output/evaluation_metrics_summary.txt"
    with open(metrics_file_path, "w", encoding="utf-8") as f:
        f.write("📋 Consolidated Evaluation Metrics:\n")
        f.write("(Note: Test metrics are shown first, followed by training metrics in brackets.)\n\n")
        metrics_table = tabulate(pd.DataFrame(results), headers="keys", tablefmt="grid")
        f.write(metrics_table + "\n\n")

        for model_name, confusion_table in confusion_matrices.items():
            f.write(f"📊 Confusion Matrix for {model_name}:\n")
            f.write(confusion_table + "\n\n")

    print(f"\n💾 Saved consolidated evaluation metrics to {metrics_file_path}")

    # 2. Save feature importance scores
    for result in feature_importance_results:
        model_name = result["Model"]
        feature_importances = pd.Series(result["Feature Importances"]).sort_values(ascending=False)
        file_path = f"output/feature_importances_{model_name.replace(' ', '_').lower()}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"📊 Feature Importance Scores for {model_name}:\n")
            f.write(feature_importances.to_string())
        print(f"💾 Saved feature importance for {model_name} to {file_path}")

    # 3. Save ROC curves
    if roc_data:
        combined_roc_file_path = "output/roc_curves_combined.png"
        create_and_save_roc_plot(roc_data, combined_roc_file_path)
        print(f"💾 Saved combined ROC curves to {combined_roc_file_path}")
    
    # 4. Save Precision-Recall curves
    if pr_data:
        combined_pr_file_path = "output/pr_curves_combined.png"
        create_and_save_pr_plot(pr_data, combined_pr_file_path)
        print(f"💾 Saved combined PR curves to {combined_pr_file_path}")


def create_and_save_roc_plot(roc_data, file_path):
    """
    Create and save ROC curve plot.
    
    Args:
        roc_data: ROC curve data
        file_path: Output file path
    """
    plt.figure(figsize=(10, 8))
    for roc in roc_data:
        plt.plot(roc["FPR"], roc["TPR"], label=f"{roc['Model']} (AUC = {roc['AUC']:.2f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier", linewidth=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - Comparison of Models", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(color="lightgray", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_and_save_pr_plot(pr_data, file_path):
    """
    Create and save Precision-Recall curve plot.
    
    Args:
        pr_data: Precision-Recall curve data
        file_path: Output file path
    """
    plt.figure(figsize=(10, 8))
    for pr in pr_data:
        plt.step(
            pr["Recall"], 
            pr["Precision"],
            where='post',
            label=f"{pr['Model']} (AP = {pr['AP']:.2f})",
            linewidth=2
        )
    
    # Add a reference line for a random classifier
    baseline = sum(pr_data[0]["Recall"] > 0) / len(pr_data[0]["Recall"])
    plt.axhline(y=baseline, color='gray', linestyle='--', label=f'Random Classifier (AP = {baseline:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves - Comparison of Models", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(color="lightgray", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()