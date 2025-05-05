# src/evaluate_models.py

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance
import seaborn as sns
from tabulate import tabulate
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    auc,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
import matplotlib.pyplot as plt


def evaluate_models(trained_models, feature_names, X_train, X_test, y_train, y_test):
    """
    Evaluate a list of trained models and save evaluation results.

    Args:
        trained_models (list): List of trained models to evaluate.
        X_train (pd.DataFrame or np.ndarray): Training feature matrix.
        X_test (pd.DataFrame or np.ndarray): Testing feature matrix.
        y_train (pd.Series or np.ndarray): Training target variable.
        y_test (pd.Series or np.ndarray): Testing target variable.

    Returns:
        dict: Dictionary containing consolidated evaluation results.
    """
    print(f"\n📊 Evaluating best models...")

    # Ensure the output directory exists
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Lists to store results
    results = []  # Evaluation metrics for each model
    roc_data = []  # ROC curve data for all models
    pr_data = []  # Precision-Recall curve data for all models

    # Loop through each trained model and evaluate
    for i, (model_name, best_model, training_time, model_size_kb) in enumerate(trained_models):
        print(f"\n   📋 Evaluating {model_name} model...")
        start_time = time.time()

        # Calculate metrics
        train_metrics = calculate_metrics(best_model, X_train, y_train)
        test_metrics = calculate_metrics(best_model, X_test, y_test)
        print(f"      └── Computing general evaluation metrics scores...")

        # Process probability-based metrics
        roc_auc, pr_auc = process_probability_metrics(model_name, best_model, X_test, y_test, roc_data, pr_data)

        # Process feature importance
        process_feature_importance(model_name, best_model, X_train, y_train, feature_names, output_dir)

        # Generate confusion matrix
        generate_confusion_matrix(model_name, best_model, X_test, y_test, output_dir)
        
        # Generate learning curves
        generate_learning_curves(model_name, best_model, X_train, y_train, output_dir)

        # Generate calibration curve
        generate_calibration_curve(model_name, best_model, X_test, y_test, output_dir)

        # Format and store results
        formatted_metrics = format_metrics(model_name, train_metrics, test_metrics, roc_auc, pr_auc)
        results.append(formatted_metrics)

        end_time = time.time()
        evaluation_time = end_time - start_time
        print(f"      └── Evaluation completed in {evaluation_time:.2f} seconds!")

        # Add evaluation_time to the trained_models list
        trained_models[i] = (model_name, best_model, training_time, model_size_kb, formatted_metrics, evaluation_time)

    # Save consolidated results
    save_consolidated_metrics(results, output_dir)
    save_roc_curves(roc_data, output_dir)
    save_pr_curves(pr_data, output_dir)
    print(f"\n💾 Saved evaluation metrics and charts to {output_dir} folder!")

    return trained_models


def calculate_metrics(model, X, y_true):
    """Calculate basic classification metrics."""
    y_pred = model.predict(X)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
    }


def process_probability_metrics(model_name, model, X_test, y_test, roc_data, pr_data):
    """Process probability-based metrics and curves."""
    roc_auc = None
    pr_auc = None
    try:
        y_test_probs = model.predict_proba(X_test)[:, 1]

        # Compute ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_test_probs)
        roc_auc = auc(fpr, tpr)
        roc_data.append({"Model": model_name, "FPR": fpr, "TPR": tpr, "AUC": roc_auc})
        print(f"      └── Computing ROC AUC data...")

        # Compute Precision-Recall curve data
        precision, recall, _ = precision_recall_curve(y_test, y_test_probs)
        pr_auc = auc(recall, precision)
        pr_data.append({"Model": model_name, "Precision": precision, "Recall": recall, "AUC-PR": pr_auc})
        print(f"      └── Computing Precision-Recall AUC data...")

    except AttributeError:
        print(f"      └── ⚠️ Model {model_name} does not support `predict_proba`. Skipping ROC and PR curve generation.")
    
    return roc_auc, pr_auc


def process_feature_importance(model_name, best_model, X_train, y_train, feature_names, output_dir):
    """Extract and save feature importance scores."""
    try:
        # Check if the model has feature importances or coefficients
        if hasattr(best_model, "coef_"):
            # Logistic Regression
            feature_importances = pd.Series(
                abs(best_model.coef_[0]),
                index=feature_names
            ).sort_values(ascending=False)
        elif hasattr(best_model, "feature_importances_"):
            # Tree-based models
            feature_importances = pd.Series(
                best_model.feature_importances_,
                index=feature_names
            ).sort_values(ascending=False)
        else:
            # Use permutation importance for other models
            perm_importance = permutation_importance(
                best_model, X_train, y_train, scoring="accuracy", n_repeats=10, random_state=42
            )
            feature_importances = pd.Series(
                perm_importance.importances_mean,
                index=feature_names
            ).sort_values(ascending=False)

        # Save feature importance to a file
        os.makedirs(f"{output_dir}/feature_importance", exist_ok=True)
        file_path = f"{output_dir}/feature_importance/feature_importances_{model_name.replace(' ', '_').lower()}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"--- 📊 Feature Importance Scores for {model_name}:---\n")
            f.write(feature_importances.to_string())
        print(f"      └── Computing and saving feature importance scores...")

    except Exception as e:
        print(f"      └── ⚠️  Could not compute or save feature importance for {model_name}: {str(e)}")


def generate_confusion_matrix(model_name, model, X_test, y_test, output_dir):
    """Generate and save confusion matrix visualization."""
    try:
        print(f"      └── Generating and plotting the confusion matrix...")
        cm = confusion_matrix(y_test, model.predict(X_test))

        # Normalize confusion matrix for visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, None]

        # Save confusion matrix plot as PNG
        os.makedirs(f"{output_dir}/confusion_matrix", exist_ok=True)
        png_filename = f"{output_dir}/confusion_matrix/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=["No-Show", "Show"], yticklabels=["No-Show", "Show"])
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.title(f'Normalized Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(png_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"      └── ⚠️ Error occurred while computing or saving confusion matrix for {model_name}: {str(e)}")


def generate_learning_curves(model_name, model, X_train, y_train, output_dir):
    """Generate and save learning curves."""
    try:
        print(f"      └── Generating and plotting learning curves...")
        train_sizes = np.linspace(0.1, 1.0, 10)  # Define training sizes
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes, cv=5, scoring='f1', n_jobs=-1
        )

        # Calculate mean and standard deviation for training and test scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="r")
        plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training score")
        plt.plot(train_sizes, test_mean, 'o-', color="r", label="Cross-validation score")

        # Add labels, title, and legend
        plt.title(f"Learning Curves - {model_name}", fontsize=14, fontweight="bold")
        plt.xlabel("Training examples", fontsize=12)
        plt.ylabel("F1 Score", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc="best", fontsize=10)
        plt.tight_layout()

        # Save the learning curve plot
        os.makedirs(f"{output_dir}/learning_curves", exist_ok=True)
        learning_curve_file_path = f"{output_dir}/learning_curves/learning_curve_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(learning_curve_file_path, dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"      └── ⚠️ Error generating learning curves for {model_name}: {str(e)}")


def generate_calibration_curve(model_name, model, X_test, y_test, output_dir):
    """Generate and save calibration curve."""
    try:
        print(f"      └── Generating and plotting calibration curve...")
        prob_pos = model.predict_proba(X_test)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

        # Plot calibration curve
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{model_name}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability", fontsize=12)
        plt.ylabel("Fraction of Positives", fontsize=12)
        plt.title(f"Calibration Curve - {model_name}", fontsize=14, fontweight="bold")
        plt.legend(loc="upper left", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save calibration curve plot
        os.makedirs(f"{output_dir}/calibration_curves", exist_ok=True)
        calibration_curve_file_path = f"{output_dir}/calibration_curves/calibration_curve_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(calibration_curve_file_path, dpi=300, bbox_inches="tight")
        plt.close()

    except AttributeError:
        print(f"      └── ⚠️ Model {model_name} does not support `predict_proba`. Skipping calibration curve generation.")
    except Exception as e:
        print(f"      └── ⚠️ Error generating calibration curve for {model_name}: {str(e)}")


def format_metrics(model_name, train_metrics, test_metrics, roc_auc, pr_auc):
    """Format metrics for reporting."""
    return {
        "Model": model_name,
        "Accuracy": f"{test_metrics['Accuracy']:.2f} ({train_metrics['Accuracy']:.2f})",
        "Precision": f"{test_metrics['Precision']:.2f} ({train_metrics['Precision']:.2f})",
        "Recall": f"{test_metrics['Recall']:.2f} ({train_metrics['Recall']:.2f})",
        "F1-Score": f"{test_metrics['F1-Score']:.2f} ({train_metrics['F1-Score']:.2f})",
        "ROC AUC": f"{roc_auc:.2f}" if roc_auc is not None else "N/A",
        "PR AUC": f"{pr_auc:.2f}" if pr_auc is not None else "N/A",
    }


def save_consolidated_metrics(results, output_dir):
    """Save consolidated evaluation metrics to file."""
    metrics_file_path = f"{output_dir}/evaluation_metrics_summary.txt"
    with open(metrics_file_path, "w", encoding="utf-8") as f:
        f.write("📋 Consolidated Evaluation Metrics:\n")
        f.write("(Note: Test metrics are shown first, followed by training metrics in brackets.)\n\n")
        metrics_table = tabulate(
            pd.DataFrame(results),
            headers="keys",
            tablefmt="grid",
            floatfmt=".2f"
        )
        f.write(metrics_table + "\n\n")


def save_roc_curves(roc_data, output_dir):
    """Save combined ROC curves to file."""
    if not roc_data:
        print("      └── ❌ No ROC data available to generate the curve.")
    else:
        combined_roc_file_path = f"{output_dir}/roc_curves_combined.png"
        plt.figure(figsize=(10, 8))
        for roc in roc_data:
            plt.plot(roc["FPR"], roc["TPR"], label=f"{roc['Model']} (AUC = {roc['AUC']:.2f})", linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier", linewidth=1.5)
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curves - Comparison of Models", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(color="lightgray", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(combined_roc_file_path, dpi=300, bbox_inches="tight")
        plt.close()


def save_pr_curves(pr_data, output_dir):
    """Save combined Precision-Recall curves to file."""
    if not pr_data:
        print("      └── ❌ No Precision-Recall data available to generate the curve.")
    else:
        combined_pr_file_path = f"{output_dir}/pr_curves_combined.png"
        plt.figure(figsize=(10, 8))
        for pr in pr_data:
            plt.plot(pr["Recall"], pr["Precision"], label=f"{pr['Model']} (AUC-PR = {pr['AUC-PR']:.2f})", linewidth=2)
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Precision-Recall Curves - Comparison of Models", fontsize=14, fontweight="bold")
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(color="lightgray", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(combined_pr_file_path, dpi=300, bbox_inches="tight")
        plt.close()


def compute_metrics(model, X, y_true):
    """
    Compute evaluation metrics for a given model and dataset.

    Args:
        model: Trained machine learning model.
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y_true (pd.Series or np.ndarray): True target values.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }