## Module to preprocess the dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## Split into features and target
def xy_split(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
  X = df.drop(columns=target_column)
  y = df[target_column]

  return X,y

## Split into training and test
def traintest_split(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

  return X_train, X_test, y_train, y_test

## Feature Scaling
def feature_scaling(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  return X_train_scaled, X_test_scaled