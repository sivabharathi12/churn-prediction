import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    df = df.copy()

    # Drop customer ID
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Encode categorical variables
    for col in df.select_dtypes(include='object').columns:
        if col != 'Churn':
            df[col] = LabelEncoder().fit_transform(df[col])

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return X, y
