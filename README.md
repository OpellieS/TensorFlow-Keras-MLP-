# TensorFlow-Keras-MLP-
TensorFlow / Keras (MLP)-IEEE-CIS Fraud Detection (Vesta)
# 💳 IEEE-CIS Fraud Detection: Regularized MLP with TensorFlow

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

This project implements a Deep Learning approach (Multi-Layer Perceptron) to detect fraudulent transactions using the IEEE-CIS Fraud Detection dataset. The pipeline focuses on handling **severe class imbalance**, **temporal data splitting**, and **advanced missing value imputation**.

## 📌 Project Overview

Fraud detection is a classic class-imbalance problem where positive cases (Fraud) are extremely rare compared to negative cases (Legitimate). This project contrasts standard training against **class-weighted learning** to optimize Recall and Precision trade-offs.

### Key Techniques Employed:
* **Time-Aware Split:** Validation data is split based on time (`TransactionDT`) to prevent look-ahead bias.
* **Advanced Imputation:** Combining "Missing Indicators" with **MICE (Multivariate Imputation by Chained Equations)** via `IterativeImputer`.
* **Regularized MLP:** Neural Network with L2 Regularization, Dropout, and Early Stopping.
* **Imbalance Handling:** Computed `class_weights` to penalize misclassification of the minority class.

## 🛠️ Data Pipeline & Preprocessing

The preprocessing pipeline is designed to be robust and reproducible (`SEED=42`).

1.  **Data Merging:** Merged `transaction` and `identity` datasets.
2.  **Subsampling:** Utilized a reproducible subset (N=150k) for computational efficiency.
3.  **Feature Engineering:**
    * **Numeric:** Standard Scaling (`StandardScaler`).
    * **Categorical:** One-Hot Encoding (`get_dummies`) with alignment between Train and Validation sets.
4.  **Missing Value Strategy:**
    * *Flagging:* Created binary columns for missing values (preserving the signal that "data is missing").
    * *Imputation:* Applied Regression-based imputation (`IterativeImputer`) for columns with significant missing rates (5% - 90%).

## 🧠 Model Architecture

The main model is a **Multi-Layer Perceptron (MLP)** built with TensorFlow/Keras:

| Layer | Type | Specifications |
| :--- | :--- | :--- |
| **Input** | Dense | 256 Units, ReLU, **L2 Regularization** |
| **Dropout** | Dropout | Rate = 0.3 |
| **Hidden** | Dense | 128 Units, ReLU |
| **Dropout** | Dropout | Rate = 0.3 |
| **Output** | Dense | 1 Unit, Sigmoid |

*Optimizer:* Adam
*Loss:* Binary Crossentropy

## 📊 Experiments & Results

We compared two training configurations to analyze the impact of handling imbalance:

1.  **Baseline MLP:** Trained without class weights.
2.  **Weighted MLP:** Trained with `class_weight="balanced"` to handle the minority class.

### Evaluation Metrics
* **ROC-AUC**
* **Precision & Recall**
* **F1-Score**
* **Confusion Matrix**

> **Note on Strategy:**
> * **Missing Values:** Missingness was treated as informative rather than noise. High-missing features were retained if they exhibited conditional predictive power.
> * **Overfitting:** Controlled via Early Stopping (patience=3) and restoring best weights.

## 🚀 Usage

### Prerequisites
```bash
pip install tensorflow pandas numpy scikit-learn seaborn matplotlib
