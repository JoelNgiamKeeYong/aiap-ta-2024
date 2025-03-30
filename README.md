# 🏨 **Hotel No-Show Prediction Pipeline**

## 😀 Author

- Full Name: **Joel Ngiam Kee Yong**
- Email: [joelngiam@yahoo.com.sg](joelngiam@yahoo.com.sg)

## 🌐 Overview

This submission predicts hotel no-shows using a configurable ML pipeline. Folder structure:

Required:

- `src/`: Python modules for data loading, preprocessing, and modeling.
- `data/`: Placeholder for `noshow.db` (not submitted).
- `eda.ipynb`: EDA notebook.
- `run.sh`: Executes the pipeline.
- `requirements.txt`: Dependencies.

Additional:

- `assets/`
- `models/`
- `READ_ABOUT_ME.md`
- `run_download_data.sh`
- `pipeline.ipynb`

## 📋 Execution Instructions

1. Place `noshow.db` in `data/`.

   - The dataset is available for download at the following URL:  
     [https://techassessment.blob.core.windows.net/aiap-pys-2/noshow.db](https://techassessment.blob.core.windows.net/aiap-pys-2/noshow.db)
   - To download the dataset in the CLI, run the script: `bash run_download_data.sh`

2. Install dependencies: `pip install -r requirements.txt`.
3. Run: `bash run.sh` (or `python src/pipeline.py` on Windows).
4. Modify `pipeline.py` config (e.g., model type) for experimentation.

## 🏭 Pipeline Flow

1. Load data from `data/noshow.db` (SQLite).
2. Preprocess (e.g., encode `country`, handle dates).
3. Train models (e.g., Logistic Regression, Random Forest).
4. Evaluate (e.g., accuracy, F1-score).

## 🔎 EDA Findings

- High no-show rates in certain `booking_month`s; influenced pipeline feature selection.

## 🛠️ Feature Processing

| Feature         | Processing       |
| --------------- | ---------------- |
| `country`       | One-hot encoded  |
| `booking_month` | Converted to int |
| `price`         | Normalized       |

## 🤖 Model Choices

- Logistic Regression: Baseline simplicity.
- Random Forest: Handles non-linearity.
- XGBoost: Boosted performance.

## 📊 Evaluation

- Metrics: Accuracy, F1-score (prioritizes no-show detection).
- Random Forest outperformed with 0.85 F1.

## ⚠️ Limitations

- asd da

## 🚀 Next Steps

- sadda
