from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def train_and_evaluate(X, y, model_type="random_forest"):
    """
    Train and evaluate a machine learning model for no-show prediction.
    
    Args:
        X (pandas.DataFrame): Features for training.
        y (pandas.Series): Target variable (no_show).
        model_type (str): Type of model to use ('random_forest', 'logistic_regression', 'xgboost').
    
    Returns:
        dict: Dictionary containing the trained model and evaluation metrics.
    """
    # Map model_type to the corresponding model
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(random_state=42),
        "xgboost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    # Select the model, default to RandomForestClassifier if model_type is invalid
    model = models.get(model_type, RandomForestClassifier(random_state=42))
    print(f"🛠️ Training {model_type} model...")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    print(f"✅ Model training completed! 🎉")
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='binary'),
        "recall": recall_score(y_test, y_pred, average='binary'),
        "f1": f1_score(y_test, y_pred, average='binary')
    }
    
    print(f"📊 Evaluation Metrics for {model_type}:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Return the trained model and evaluation metrics
    return {
        "model": model,
        "metrics": metrics
    }

if __name__ == "__main__":
    # Example usage (for testing purposes)
    import pandas as pd
    # Dummy data for testing
    X = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50]
    })
    y = pd.Series([0, 1, 0, 1, 0])
    
    result = train_and_evaluate(X, y, model_type="random_forest")
    print("Test Results:", result["metrics"])