# 📊 Customer Churn Prediction (Advanced ML Model)

## 🚀 Overview
This project predicts customer churn using Machine Learning.  
It uses an advanced pipeline with preprocessing, SMOTE, and XGBoost.

---

## ⚙️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)

---

## 🔍 Model Features
- Data preprocessing pipeline
- Missing value handling (Imputation)
- One-hot encoding
- Feature scaling
- SMOTE for class imbalance
- Hyperparameter tuning (GridSearchCV)
- Custom threshold (0.4)

---

## 📈 Results
- Recall (Churn): **82%**
- ROC-AUC: **0.84**
- Accuracy: **74%**

---

## 🧠 Business Insight
The model prioritizes recall to identify maximum churners, which is critical in retention strategies.

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python -m src.train