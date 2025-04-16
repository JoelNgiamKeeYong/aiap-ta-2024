import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.inspection import permutation_importance
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


def train_and_evaluate_models(
        models,
        X_train, X_test, y_train, y_test,
        n_jobs, cv_folds, scoring_metric, feature_names=None
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
    # Ensure the models and output directory exists
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # Lists to store results
    results = []  # Evaluation metrics for each model
    roc_data = []  # ROC curve data for all models
    pr_data = []  # Precision-Recall curve data for all models

    # Loop through each model and perform training and evaluation
    for model_name, model_info in models.items():
        # Train the model with hyperparameter tuning
        best_model = train_model(
            model=model_info["model"],
            params=model_info["params"],
            X_train=X_train,
            y_train=y_train,
            scoring_metric=scoring_metric,
            cv_folds=cv_folds,
            n_jobs=n_jobs,
            model_name=model_name
        )

        # Evaluate the model
        evaluate_model(
            best_model=best_model, 
            model_name=model_name, 
            X_train=X_train, 
            X_test=X_test,
            y_train=y_train, 
            y_test=y_test,
            results=results, 
            roc_data=roc_data, 
            pr_data=pr_data, 
            feature_names=feature_names
        )      

    # Save consolidated results
    save_all_results(
        results, 
        roc_data, 
        pr_data
    )


def train_model(model, params, X_train, y_train, scoring_metric, cv_folds, n_jobs, model_name):
    """
    Train a model with hyperparameter tuning using GridSearchCV and save the trained model.

    Args:
        model: The machine learning model to tune.
        params (dict): Hyperparameter grid for the model.
        X_train (pandas.DataFrame or numpy.ndarray): Training feature matrix.
        y_train (pandas.Series or numpy.ndarray): Training target variable.
        scoring_metric (str): Scoring metric for GridSearchCV.
        cv_folds (int): Number of cross-validation folds.
        n_jobs (int): Number of parallel jobs for GridSearchCV.
        model_name (str): Name of the model being trained.

    Returns:
        object: Best model after hyperparameter tuning.
    """
    print(f"\n🛠️  Training {model_name} model...")
    start_time = time.time()

    # Perform hyperparameter tuning using GridSearchCV
    print(f"   └── Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring=scoring_metric,
        cv=cv_folds,
        n_jobs=n_jobs
    )
    grid_search.fit(X_train, y_train)

    # Extract the best model and parameters
    best_model = grid_search.best_estimator_
    print(f"   └── Best parameters: {grid_search.best_params_}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"✅ Training completed successfully in {elapsed_time:.2f} seconds!")

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_name.replace(' ', '_').lower()}_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"💾 Saving trained model to {model_path}...")

    return best_model


def evaluate_model(best_model, model_name, X_train, X_test, y_train, y_test, results, roc_data, pr_data, feature_names):
    """
    Evaluate the trained model and compute metrics, ROC/PR curves, feature importance, and save results.

    Args:
        best_model: The best model after hyperparameter tuning.
        model_name (str): Name of the model being evaluated.
        X_train (pandas.DataFrame or numpy.ndarray): Training feature matrix.
        X_test (pandas.DataFrame or numpy.ndarray): Testing feature matrix.
        y_train (pandas.Series or numpy.ndarray): Training target variable.
        y_test (pandas.Series or numpy.ndarray): Testing target variable.
        results (list): List to store evaluation metrics for each model.
        roc_data (list): List to store ROC curve data for all models.
        pr_data (list): List to store Precision-Recall curve data for all models.
        feature_names (list): List of feature names (optional, for sparse matrices).
    """
    print(f"📊 Evaluating {model_name} model...")
    start_time = time.time()

    # Compute evaluation metrics
    def calculate_metrics(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred),
        }

    train_metrics = calculate_metrics(y_train, best_model.predict(X_train))
    test_metrics = calculate_metrics(y_test, best_model.predict(X_test))
    print(f"   └── Computing general evaluation metrics scores...")

    # Handle probability-based metrics (ROC, PR curves)
    roc_auc = None
    pr_auc = None
    try:
        y_test_probs = best_model.predict_proba(X_test)[:, 1]

        # Compute ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_test_probs)
        roc_auc = auc(fpr, tpr)
        roc_data.append({"Model": model_name, "FPR": fpr, "TPR": tpr, "AUC": roc_auc})
        print(f"   └── Computing ROC AUC data...")

        # Compute Precision-Recall curve data
        precision, recall, _ = precision_recall_curve(y_test, y_test_probs)
        pr_auc = auc(recall, precision)
        pr_data.append({"Model": model_name, "Precision": precision, "Recall": recall, "AUC-PR": pr_auc})
        print(f"   └── Computing Precision-Recall AUC data...")

    except AttributeError:
        print(f"   └── ⚠️ Model {model_name} does not support `predict_proba`. Skipping ROC and PR curve generation.")

    # Extract and save feature importance scores
    try:
        if hasattr(best_model, 'feature_importances_'):
            feature_importances = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=False)
        else:
            # Use permutation importance for models like Logistic Regression
            perm_importance = permutation_importance(best_model, X_train, y_train, scoring="accuracy", n_repeats=10, random_state=42)
            feature_importances = pd.Series(perm_importance.importances_mean, index=feature_names).sort_values(ascending=False)

        # Save feature importance to a file
        os.makedirs("output/feature_importance", exist_ok=True)
        file_path = f"output/feature_importance/feature_importances_{model_name.replace(' ', '_').lower()}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"📊 Feature Importance Scores for {model_name}:\n")
            f.write(feature_importances.to_string())
        print(f"   └── Computing and saving feature importance scores...")

    except Exception as e:
        print(f"   └── ⚠️ Could not compute or save feature importance for {model_name}: {str(e)}")

    # Compute and save confusion matrix
    try:
        cm = confusion_matrix(y_test, best_model.predict(X_test))

        # Normalize confusion matrix for visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, None]

        # Save confusion matrix plot as PNG
        os.makedirs("output/confusion_matrix", exist_ok=True)
        png_filename = f"output/confusion_matrix/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=["No-Show", "Show"], yticklabels=["No-Show", "Show"])
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.title(f'Normalized Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(png_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   └── Generating and plotting the confusion matrix...")

    except Exception as e:
        print(f"   ⚠️ Error occurred while computing or saving confusion matrix for {model_name}: {str(e)}")
    
    # Generate and save learning curves
    try:
        print(f"   └── Generating learning curves for...")
        train_sizes = np.linspace(0.1, 1.0, 10)  # Define training sizes
        train_sizes, train_scores, test_scores = learning_curve(
            best_model, X_train, y_train,
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
        os.makedirs("output/learning_curves", exist_ok=True)
        learning_curve_file_path = f"output/learning_curves/learning_curve_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(learning_curve_file_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"   └── Saved learning curves for {model_name} to {learning_curve_file_path}")

    except Exception as e:
        print(f"   ⚠️ Error generating learning curves for {model_name}: {str(e)}")

    # Combine test and training metrics into a single formatted string for each metric
    formatted_metrics = {
        "Model": model_name,
        "Accuracy": f"{test_metrics['Accuracy']:.2f} ({train_metrics['Accuracy']:.2f})",
        "Precision": f"{test_metrics['Precision']:.2f} ({train_metrics['Precision']:.2f})",
        "Recall": f"{test_metrics['Recall']:.2f} ({train_metrics['Recall']:.2f})",
        "F1-Score": f"{test_metrics['F1-Score']:.2f} ({train_metrics['F1-Score']:.2f})",
        "ROC AUC": f"{roc_auc:.2f}" if roc_auc is not None else "N/A",
        "PR AUC": f"{pr_auc:.2f}" if pr_auc is not None else "N/A",
    }
    results.append(formatted_metrics)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"✅ Evaluation completed successfully in {elapsed_time:.2f} seconds!")


def save_all_results(results, roc_data, pr_data):
    """
    Save all consolidated results: evaluation metrics summary, combined ROC curve, and combined PR curve.

    Args:
        results (list): List of evaluation metrics for each model.
        roc_data (list): List of ROC curve data for all models.
        pr_data (list): List of Precision-Recall curve data for all models.
    """
    # 1. Save consolidated evaluation metrics summary to a .txt file
    metrics_file_path = "output/evaluation_metrics_summary.txt"
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
    print(f"\n💾 Saved consolidated evaluation metrics to {metrics_file_path}")

    # 2. Create and save a combined ROC curve as a .png file
    if not roc_data:
        print("❌ No ROC data available to generate the curve.")
    else:
        combined_roc_file_path = "output/roc_curves_combined.png"
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
        print(f"💾 Saved combined ROC curves to {combined_roc_file_path}")

    # 3. Create and save a combined Precision-Recall curve as a .png file
    if not pr_data:
        print("❌ No Precision-Recall data available to generate the curve.")
    else:
        combined_pr_file_path = "output/pr_curves_combined.png"
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
        print(f"💾 Saved combined Precision-Recall curves to {combined_pr_file_path}")
