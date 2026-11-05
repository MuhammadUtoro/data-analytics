## Test file for transform module
import pandas as pd
import pytest
from src.transform import *

## Test remove_current() - Here we need integration test
def test_remove_current():
  ## Arrange
  test_data = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6],
    "grade": ["A", "A", "C", "D", "E", "B"],
    "sub_grade": ["A1", "D2", "C3", "B5", "E1", "F2"],
    "emp_length": ["< 1 year", "2 years", "3 years", "4 years", "10+ years", "5 years"],
    "loan_status": ["Current", "Fully Paid", "Current", "Charged Off", "Fully Paid", "Charged Off"]
  })

  ## Act
  df = remove_current(test_data)

  ## Assert
  assert "Current" not in df["loan_status"].values
  assert len(df) == 4
  assert df["loan_status"].value_counts()["Fully Paid"] == 2
  assert df["loan_status"].value_counts()["Charged Off"] == 2

## Test for ordinal mapping - grade
def test_ordinal_mapping_grade():
  ## Arrange
  test_data = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6],
    "grade": ["A", "A", "C", "D", "E", "B"],
    "sub_grade": ["A1", "D2", "C3", "B5", "E1", "F2"],
    "emp_length": ["< 1 year", "2 years", "3 years", "4 years", "10+ years", "5 years"],
    "loan_status": ["Current", "Fully Paid", "Current", "Charged Off", "Fully Paid", "Charged Off"]
  })

  ## Act
  df = map_ordinal(test_data, "grade")

  ## Assert
  expected_values = [6, 6, 4, 3, 2, 5]
  assert df["grade"].tolist() == expected_values

## Test for ordinal mapping - sub_grade
def test_ordinal_mapping_subgrade():
  ## Arrange
  test_data = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6],
    "grade": ["A", "A", "C", "D", "E", "B"],
    "sub_grade": ["A1", "D2", "C3", "B5", "E1", "F2"],
    "emp_length": ["< 1 year", "2 years", "3 years", "4 years", "10+ years", "5 years"],
    "loan_status": ["Current", "Fully Paid", "Current", "Charged Off", "Fully Paid", "Charged Off"]
  })

  ## Act
  df = map_ordinal(test_data, "sub_grade")

  ## Assert
  expected_values = [34, 18, 22, 25, 14, 8]
  assert df["sub_grade"].tolist() == expected_values

## Test for ordinal mapping - emp_length
def test_ordinal_mapping_emp_length():
  ## Arrange
  test_data = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6],
    "grade": ["A", "A", "C", "D", "E", "B"],
    "sub_grade": ["A1", "D2", "C3", "B5", "E1", "F2"],
    "emp_length": ["< 1 year", "2 years", "3 years", "4 years", "10+ years", "5 years"],
    "loan_status": ["Current", "Fully Paid", "Current", "Charged Off", "Fully Paid", "Charged Off"]
  })

  ## Act
  df = map_ordinal(test_data, "emp_length")

  ## Assert
  expected_values = [0, 2, 3, 4, 10, 5]
  assert df["emp_length"].tolist() == expected_values