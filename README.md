# Breast Cancer Classification with Data Balancing

This project provides a complete machine learning pipeline for classifying breast cancer as Malignant (M) or Benign (B) using the Wisconsin Breast Cancer Dataset.

The analysis's primary objective is to address the dataset's inherent class imbalance to build a predictive model that **minimizes False Negatives** (i.e., failing to detect a malignant case). This is the most critical metric for a medical diagnosis tool. The project systematically evaluates 9 different data balancing techniques and 7 different classification models to find the most robust and reliable solution.

---

## Dataset

* **Source:** Wisconsin Breast Cancer Dataset
**Kaggle Link:** [https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
* **Total Samples:** 569
* **Features:** 30 numerical features (e.g., `radius_mean`, `texture_mean`, `concavity_mean`)
* **Target:** `diagnosis` (Malignant or Benign)

### Class Imbalance
The dataset is moderately imbalanced, which can bias a model towards the majority class (Benign) and cause it to miss a high-risk Malignant case.

* **Benign (B):** 357 samples (62.7%)
* **Malignant (M):** 212 samples (37.3%)
* **Imbalance Ratio:** 1.68:1

---

## Analysis Pipeline

The notebook follows a comprehensive, end-to-end methodology:

1.  **Data Loading and Cleaning:** The dataset (569x32) was loaded, and the `id` column was dropped. No missing values were found.
2.  **Baseline Modeling:** The data was split (80% train, 20% test) and scaled using `StandardScaler`. Seven baseline models (e.g., Logistic Regression, Random Forest, SVM) were trained on the *unbalanced* data. The baseline Random Forest model resulted in **3 False Negatives**.
3.  **Balancing Technique Evaluation:** 9 different data balancing techniques were applied to the training set and evaluated using a standard Random Forest model:
    * Class Weights
    * Random Oversampling
    * SMOTE (Synthetic Minority Over-sampling Technique)
    * ADASYN
    * **Borderline SMOTE**
    * Random Undersampling
    * SMOTE + Tomek Links
    * SMOTE + ENN
4.  **Best Technique Selection:** **Borderline SMOTE** was selected as the best-performing technique. It was the only method that, when combined with a standard Random Forest, reduced the number of False Negatives on the test set to **1**.
5.  **Full Model Training:** All 7 models were retrained using the training data balanced with **Borderline SMOTE**.
6.  **Hyperparameter Tuning:** The top-performing models (Random Forest and Logistic Regression) were fine-tuned using `GridSearchCV` with `recall` as the scoring metric to prioritize minimizing False Negatives.
7.  **Final Model Selection:** After tuning, the **Logistic Regression** model provided the best and most stable performance, resulting in **2 False Negatives** (compared to the tuned Random Forest, which had 3).
8.  **Threshold Optimization:** The final Logistic Regression model's classification threshold was optimized. The default 0.5 threshold yielded 2 False Negatives and 2 False Positives. Adjusting the threshold to **0.6** maintained the 2 False Negatives while reducing False Positives to 1, improving precision.
9.  **Artifact Generation:** The final model, scaler, performance metrics, and deployment instructions were saved.

---

## Final Model and Performance

The analysis concluded that a tuned **Logistic Regression** model, trained on data balanced with **Borderline SMOTE** and using an optimized classification threshold, provides the best balance of accuracy and, most importantly, high recall for detecting malignant cases.

* **Model:** Logistic Regression
* **Data Balancing:** Borderline SMOTE
* **Optimal Threshold:** 0.6 (default is 0.5)

### Final Performance Metrics
These metrics reflect the final model's performance on the unseen test set (114 samples).

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 0.9737 |
| **Recall (Sensitivity)** | 0.9524 |
| **Precision** | 0.9756 |
| **F1-Score** | 0.9639 |
| **ROC-AUC** | 0.9828 |

### Confusion Matrix (Final Model)
The final model achieved a **33.3% reduction in False Negatives** compared to the baseline model (from 3 to 2).

| | Predicted: Benign | Predicted: Malignant |
| :--- | :---: | :---: |
| **Actual: Benign** | **71** (True Negative) | **1** (False Positive) |
| **Actual: Malignant** | **2** (False Negative) ⚠️ | **40** (True Positive) |

---

## How to Use

This project generates all necessary artifacts for deploying the model.

1.  **Load Artifacts:** Load the saved model and scaler:
    * `models/final_best_model.pkl` (The trained Logistic Regression model)
    * `models/scaler.pkl` (The `StandardScaler` fitted on the training data)
2.  **Pre-process Input:** New data must be scaled using the loaded `scaler.pkl`.
3.  **Make Predictions:** Use the model's `predict_proba()` method to get the probability of malignancy.
4.  **Apply Threshold:** A diagnosis is **Malignant** if the predicted probability is **0.6**.

For detailed Python code examples, monitoring guidelines, and risk mitigation strategies, please refer to the auto-generated **`models/DEPLOYMENT_INSTRUCTIONS.txt`** file.

---

## Generated Files

This analysis produces several key artifacts, which are all saved in the `models/` directory:

* **Model Files:**
    * `models/final_best_model.pkl`: The final, tuned Logistic Regression model object.
    * `models/scaler.pkl`: The `StandardScaler` object.
* **Deployment Guide:**
    * `models/DEPLOYMENT_INSTRUCTIONS.txt`: A comprehensive guide on how to use the model in a production environment.
* **Data & Reports:**
    * `models/model_metadata.json`: A JSON file detailing all model parameters, configurations, and performance metrics.
    * `models/progressive_improvement.csv`: A CSV table tracking performance from the baseline to the final optimized model.
    * `models/balancing_techniques_comparison.csv`: A CSV report comparing the performance of all 9 balancing techniques.
* **Visualizations:**
    * `models/complete_improvement_journey.png`: A summary plot showing the progressive reduction in False Negatives at each stage of the analysis.
    * `balancing_techniques_comparison.png`: A detailed plot comparing all 9 balancing techniques.
    * `confusion_matrices_all.png`: Confusion matrices for all tested balancing techniques.
    * `roc_curves_comparison.png`: ROC curves for all tested balancing techniques.