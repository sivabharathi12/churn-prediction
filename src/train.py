# ============================================================
# ADVANCED CHURN PREDICTION (CLEAN VERSION)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer   # 👈 ADD THIS IMPORT AT TOP

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier

# ============================================================
# LOAD DATA
# ============================================================

def load_data(path):
    return pd.read_csv(path)

# ============================================================
# BUILD PREPROCESSOR
# ============================================================


def build_preprocessor(df):

    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    # ✅ FIXED pipelines (handles NaN)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    return preprocessor, X, y

# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(data_path):

    print("🔹 Loading data...")
    df = load_data(data_path)

    # Preprocessing
    preprocessor, X, y = build_preprocessor(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Pipeline
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', XGBClassifier(eval_metric='logloss', random_state=42))
    ])

    # Hyperparameters
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [3, 5],
        'model__learning_rate': [0.05, 0.1]
    }

    print("🔹 Running GridSearch...")

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='recall',   # good for churn (focus on catching churners)
        n_jobs=-1
    )

    # Train
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("\n✅ Best Parameters:", grid.best_params_)

    # Predictions
    y_probs = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= 0.4).astype(int)  # custom threshold

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_pred))

    print("\nROC-AUC:", roc_auc_score(y_test, y_probs))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(best_model, "advanced_model.pkl")
    print("\n💾 Model saved as advanced_model.pkl")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    train_model("data/telco.csv")