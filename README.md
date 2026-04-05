# 📊 Customer Churn Prediction

## 🚀 Overview

This project builds a machine learning model to predict customer churn using an end-to-end pipeline.
The objective is to help businesses proactively identify customers at risk of leaving and take targeted retention actions.

---

## 🧠 Problem Statement

Customer churn significantly impacts revenue. Retaining existing customers is more cost-effective than acquiring new ones.
This model predicts whether a customer will churn (**1**) or not (**0**) based on historical data.

---

## ⚙️ Tech Stack

* **Python**
* **Pandas, NumPy** (Data handling)
* **Scikit-learn** (Preprocessing & evaluation)
* **XGBoost** (Modeling)
* **Imbalanced-learn (SMOTE)** (Handling class imbalance)

---

## 🔧 Model Pipeline

The solution uses a robust ML pipeline with:

* Data Cleaning & Preprocessing
* Missing Value Imputation
* One-Hot Encoding (Categorical features)
* Feature Scaling (StandardScaler)
* SMOTE (Class imbalance handling)
* XGBoost Classifier
* Hyperparameter Tuning (GridSearchCV)
* Custom Decision Threshold (**0.4**)

---

## 📈 Model Performance

| Metric   | Score      |
| -------- | ---------- |
| Accuracy | 74%        |
| Recall   | **82% 🔥** |
| ROC-AUC  | 0.84       |

---

## 📊 Confusion Matrix

```
[[732 303]
 [ 67 307]]
```

---

## 💡 Business Insight

The model is optimized for **high recall**, ensuring that the majority of potential churners are identified.
Although this increases false positives, it is acceptable in real-world scenarios where missing a churner is more costly than targeting a non-churner.

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python -m src.train
```

---

## 📁 Project Structure

```
churn-prediction/
│── data/
│── src/
│   └── train.py
│── advanced_model.pkl
│── requirements.txt
│── README.md
```

---

