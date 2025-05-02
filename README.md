# 🏨 **Hotel No-Show Prediction ML Pipeline**

<img src="assets/pipeline_banner.png" alt="Pipeline Banner" width="80%"/>

## 😀 Author

- Full Name: **Joel Ngiam Kee Yong**
- Email: [joelngiam@yahoo.com.sg](joelngiam@yahoo.com.sg)

## Problem Statement

asdadas

## 🌐 Overview

This project predicts hotel no-shows using a modular and configurable machine learning pipeline. The pipeline is designed to be **reusable**, **interpretable**, and **scalable** for experimentation with different models and preprocessing steps.

Folder Structure:

```
├── archives/              # Folder for inactive files and logs
├── assets/                # Images or visualization assets
├── data/                  # Location for datasets; auto-generated
├── models/                # Saved preprocessor and trained models; auto-generated
├── output/                # Results of model evaluations; auto-generated
├── src/                   # Python modules for the ML pipeline
│   └── utils/             # Utility functions for general EDA
│   └── pipeline.py        # Main executable for the pipeline
├── config.yaml            # Configuration file for the ML pipeline
├── eda.ipynb              # Exploratory Data Analysis (EDA) notebook
├── README.md              # This file
├── READ_ABOUT_ME.md       # Additional documentation
├── requirements.txt       # Dependencies for the project
└── reset.sh               # Bash script to reset the project
└── run.sh                 # Bash script to execute the entire ML pipeline
```

## 📋 Execution Instructions

1. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

2. [Optional] Manually download and place the dataset file into the `data/` folder. The following step does this automatically. Link for download: [https://techassessment.blob.core.windows.net/aiap-pys-2/noshow.db](https://techassessment.blob.core.windows.net/aiap-pys-2/noshow.db)

3. Run the ML pipeline by executing the bash script

   ```bash
   bash run.sh  # Run the full ML pipeline
   bash run.sh --lite  # Run the pipeline in lite mode, for quick debugging of pipeline
   ```

4. Experiment with the ML pipeline by modifying the configuration in `config.yaml`

5. [Optional] Reset the project

   ```bash
   bash reset.sh
   ```

## 🏭 Pipeline Flow

The pipeline consists of the following logical steps:

1. Data Loading : Load data from data/noshow.db using SQLite.
2. Data Cleaning : Handle missing values, duplicates, and inconsistencies.

Train Test Split

You should perform the train-test split before feature engineering to prevent data leakage and ensure that your model evaluation is unbiased. This approach aligns with best practices in machine learning and will help you build a robust and reliable predictive model for your hotel no-show prediction task. As we are building a predictive model and want to test on unseen data, to mimic production next time, we should split it before extensive EDA, so that when we evaluate the model, we can evaluate it on the unseen test data set. However, this may mean that some information from the test set may not be captured in the model. Despite this, a model that performs well on unseen data is more valuable than one that perfectly fits the training data but fails in production.

A stratified split ensures that the proportion of classes in the target variable (y) is preserved in both the training and testing sets. This is particularly useful for imbalanced datasets.

Used 20% train test split, so sufficient data in training and testing. Also need quite a bit for training as we doing cross-validation as well

Use grid search CV (instead of randomized search cv, why?) --> Do option to change randomized search to grid search in config, as randomized search more computationally efficient and use it first to run multiple runs to see what is the best and exploring what are the best parameters, then use grid search to fine tune

Why use normalisation or min-max scaler

Why never apply oversampling / undersampling

3. Feature Engineering : Preprocess features (e.g., encode categorical variables, normalize numerical features).
4. Model Training : Train multiple models (e.g., Logistic Regression, Random Forest, XGBoost).
5. Model Evaluation : Evaluate models using metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## 🔎 EDA Findings

- EDA is carried out in the following process:

[Insert-image]

- Rather than dedicating separate sections to univariate, bivariate, and multivariate analyses, I have integrated these tools into broader, purpose-driven sections within the exploratory data analysis (EDA) pipeline. This approach reflects the iterative nature of EDA—univariate analysis might identify missing values or outliers during data cleaning, while later revisiting it after feature engineering to validate new features. Organizing the analysis into sections like data cleaning. comprehensive exploration and feature engineering ensures a logical, coherent workflow that aligns with the progression of an end-to-end machine learning pipeline.

- This structure avoids redundancy and enhances clarity by focusing on practical outcomes, such as improving model performance or addressing business needs. Readers can easily follow the progression of the analysis, as each section has a clear purpose, demonstrating both depth and rigor without unnecessary repetition.

- High no-show rates in certain `booking_month`s; influenced pipeline feature selection.

## 🛠️ Feature Processing

| Feature          | Source     | Processing                   |
| ---------------- | ---------- | ---------------------------- |
| `booking_id`     | Original   | Dropped                      |
| `no_show`        | Original   | Target, unchanged            |
| `branch`         | Original   | Converted to `category` type |
| `booking_month`  | Original   | Converted to `int` type      |
| `arrival_month`  | Original   | Converted to int             |
| `arrival_day`    | Original   | Converted to int             |
| `checkout_month` | Original   | Converted to int             |
| `checkout_day`   | Original   | Converted to int             |
| `country`        | Original   | One-hot encoded              |
| `first_time`     | Original   | One-hot encoded              |
| `room`           | Original   | One-hot encoded              |
| `price`          | Original   | Normalized                   |
| `platform`       | Original   | One-hot encoded              |
| `num_adults`     | Original   | One-hot encoded              |
| `num_children`   | Original   | One-hot encoded              |
| `currency_type`  | Engineered | One-hot encoded              |

Why use Standard Scaler over MinMax Scaler
Why use OneHotEncoding over Ordinal Encoder

Why never use sklearn pipeline function to combine preprocessing and training? Instead using separate functions

Why use GridSearch CV and not Randomized Search?

## 🤖 Candidate Models

The selection of machine learning models for predicting customer no-shows was guided by the following key considerations:

**Nature of the Dataset**:

- **Structured/Tabular Data**: The dataset consists of structured/tabular data with a mix of numerical, categorical, and ordinal features. Tree-based models such as **XGBoost**, **Random Forest**, and **LightGBM** are highly effective for such datasets due to their ability to handle mixed data types and capture complex relationships.

- **Binary Classification**: The target variable (`no_show`) is binary (0 = Show, 1 = No-Show), making classification algorithms like **Logistic Regression**, **Random Forest**, and **XGBoost** particularly suitable.

- **Dataset Size**: With over 70,000 records, scalability is crucial. Models like **XGBoost** and **LightGBM** are optimized for large datasets and can efficiently handle high-dimensional data.

**Selected Models**:

- **Logistic Regression**: Logistic Regression is a simple and interpretable model that works well for binary classification problems. It is computationally efficient and scales well with large datasets and provides a baseline performance to compare against more complex models. Additionally, feature importance can be derived from the coefficients, which helps in understanding the impact of each feature on the target variable.

- **Random Forest**: Random Forest is an ensemble method that builds multiple decision trees and aggregates their predictions. It handles non-linear relationships and interactions between features effectively and is robust to overfitting due to bagging (bootstrap aggregating) and random feature selection. Random Forest works well with both numerical and categorical features and provides feature importance scores, which are useful for identifying key predictors. However, its training time can be longer compared to simpler models, especially with large datasets or deep trees, and it is less interpretable than Logistic Regression.

- **XGBoost**: XGBoost is a gradient boosting algorithm that builds trees sequentially, optimizing for errors made by previous trees. It performs exceptionally well on structured/tabular data and supports regularization (L1/L2 penalties), which helps prevent overfitting. XGBoost is faster than Random Forest for large datasets due to its optimized implementation and ability to handle sparse data. It also handles imbalanced datasets well and provides feature importance scores and SHAP values for interpretability. While it scales well with large datasets, it is slightly less interpretable than Logistic Regression.

- **LightGBM**: LightGBM is another gradient boosting framework that is optimized for speed and memory efficiency, especially on large datasets. It uses novel techniques such as Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) to reduce training time without sacrificing accuracy. LightGBM handles categorical features natively and efficiently manages large datasets with millions of rows and high-dimensional feature spaces. Like XGBoost, it supports regularization (L1/L2 penalties) to prevent overfitting and provides feature importance scores and SHAP values for interpretability. LightGBM is often faster than XGBoost due to its histogram-based approach and optimized tree construction. However, like other advanced models, it is slightly less interpretable than Logistic Regression.

**Why not other models?**:

- **Neural Networks**: Neural networks are not ideal for this type of structured/tabular data unless the dataset has complex patterns or unstructured components, such as text or images. For this problem, simpler models like XGBoost and LightGBM will likely outperform neural networks with less computational overhead.

- **Support Vector Machines (SVM)**: SVMs are computationally expensive for large datasets and may not scale well to 70k+ records. While effective for small to medium-sized datasets with clear decision boundaries, SVMs are not practical for this use case.

- **K-Nearest Neighbors (KNN)**: KNN is also computationally expensive for large datasets, and its performance degrades with high-dimensional data (curse of dimensionality). KNN computes distances between the query point and all training points, making it computationally expensive as the dataset grows. Training time increases significantly with larger datasets, especially if there are many features.

**Model Selection Summary**:

In summary, Logistic Regression serves as a baseline model to establish a performance benchmark, while Random Forest , XGBoost , and LightGBM are chosen for their ability to handle complex relationships and large datasets efficiently. Models like XGBoost and LightGBM are prioritized for their scalability and optimization for large datasets, ensuring computational efficiency without compromising performance. While advanced models like XGBoost and LightGBM provide excellent predictive performance, interpretability is maintained through feature importance scores and SHAP values, allowing for a deeper understanding of the factors driving predictions.

## 📊 Evaluation

The choice of evaluation metrics is critical in ensuring that the models developed are aligned with the goals of the no-show prediction task. Given the nature of the dataset and the problem at hand—predicting whether a customer will not show up for their reservation—we prioritized metrics that balance accuracy, interpretability, and fairness, especially in the context of class imbalance.

1. Accuracy

   Accuracy measures the proportion of correctly predicted instances (both "Show" and "No-Show") out of the total predictions. While it provides a general sense of model performance, it can be misleading in cases of imbalanced datasets, where one class (e.g., "Show") dominates the other. In our case, since the dataset contains more "Show" instances than "No-Show," relying solely on accuracy might overestimate the model's effectiveness. Nevertheless, accuracy remains a useful baseline metric to assess overall correctness.

2. F1 Score

   The F1-score was chosen as the primary metric because it strikes a balance between precision and recall , making it particularly suitable for imbalanced datasets.

   - Precision : Measures the proportion of correctly predicted "No-Show" instances out of all instances predicted as "No-Show." High precision ensures that when the model predicts a no-show, it is likely correct, minimizing false alarms.

   - Recall : Measures the proportion of correctly predicted "No-Show" instances out of all actual "No-Show" instances. High recall ensures that the model captures most of the true no-shows, reducing missed opportunities for intervention.

   The F1-score is the harmonic mean of precision and recall, providing a single value that reflects the model's ability to accurately detect no-shows without overly favoring either precision or recall. This is crucial for our problem, as failing to predict a no-show (false negative) could result in lost revenue or operational inefficiencies, while incorrectly predicting a no-show (false positive) might lead to unnecessary resource allocation.

3. ROC-AUC

   The Receiver Operating Characteristic (ROC) curve and its corresponding Area Under the Curve (AUC) were also used to evaluate model performance. The ROC curve plots the trade-off between the True Positive Rate (TPR) and the False Positive Rate (FPR) across different probability thresholds. ROC-AUC provides a comprehensive view of the model's ability to distinguish between "Show" and "No-Show" classes across all possible thresholds. A higher AUC indicates better overall discrimination, making it a robust metric for comparing models. Unlike accuracy, ROC-AUC is less sensitive to class imbalance, making it a reliable supplementary metric for our imbalanced dataset.

4. Confusion Matrix

   The confusion matrix provides a detailed breakdown of the model's predictions into four categories:

   - True Positives (TP): Correctly predicted no-shows.
   - True Negatives (TN): Correctly predicted shows.
   - False Positives (FP): Incorrectly predicted no-shows (Type I error).
   - False Negatives (FN): Incorrectly predicted shows (Type II error).

   This granular view helps identify specific weaknesses in the model, such as whether it struggles more with false positives or false negatives. For instance, if the cost of missing a no-show (FN) is higher than incorrectly flagging a show as a no-show (FP), we can adjust the decision threshold accordingly.

Use feature importance instea of SHAP values as calculating SHAP values is computationally very expensive and this problem does not need that much interpretability. But if required can add that in.

Given the business context of no-show prediction, the following considerations guided the choice of metrics:

Imbalanced Dataset : With fewer "No-Show" instances compared to "Show," metrics like accuracy alone would be insufficient. Instead, F1-score and ROC-AUC provide a more nuanced evaluation of model performance.
Operational Impact : Missing a no-show (FN) can have significant consequences, such as unutilized resources or lost revenue. Therefore, recall is prioritized to ensure that the model captures as many true no-shows as possible.
Balancing Precision and Recall : While high recall is desirable, excessively low precision (too many false positives) could lead to wasted efforts in mitigating non-existent no-shows. The F1-score ensures a balanced approach, optimizing both precision and recall.

## ⚠️ Limitations

- The dataset may contain synthetic features that require further verification.
- Class imbalance in the target variable (no_show) may bias the model toward the majority class.
- Feature engineering could be further refined to capture additional patterns.

## 🚀 Next Steps

- Hyperparameter Tuning : Optimize model parameters using GridSearchCV or Bayesian Optimization.
- Advanced Feature Engineering : Incorporate interaction terms or domain-specific features.
- Deployment : Deploy the best-performing model as a REST API for real-time predictions.
- Monitoring : Track model performance in production to detect drift and retrain as needed.

## Troubleshooting

1. Reset the project by running the script above and try again

## FAQ

1.
