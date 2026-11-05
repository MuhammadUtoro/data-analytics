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

## Test remove unrelated_features
def test_remove_unrelated_colummns():
  ## Arrange
  test_data = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6],
    "grade": ["A", "A", "C", "D", "E", "B"],
    "application_type": ["test01", "test02", "test03", "test04", "test05", "test06"],
    "emp_title": ["title_test01", "title_test02", "title_test03", "title_test04", "title_test05", "title_test06"],
    "sub_grade": ["A1", "D2", "C3", "B5", "E1", "F2"],
    "emp_length": ["< 1 year", "2 years", "3 years", "4 years", "10+ years", "5 years"],
    "loan_status": ["Current", "Fully Paid", "Current", "Charged Off", "Fully Paid", "Charged Off"]
  })

  drop_cols = ["id", "application_type", "emp_title"]

  ## Act
  df = remove_unrelated_columns(test_data, columns=drop_cols)

  ## Assert
  assert set(drop_cols).isdisjoint(df.columns)

## Test the one-hot encoding
def test_one_hot_encoding():
  ## Arrange
  test_data = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6],
    "grade": ["A", "A", "C", "D", "E", "B"],
    "application_type": ["test01", "test02", "test03", "test04", "test05", "test06"],
    "emp_title": ["title_test01", "title_test02", "title_test03", "title_test04", "title_test05", "title_test06"],
    "sub_grade": ["A1", "D2", "C3", "B5", "E1", "F2"],
    "emp_length": ["< 1 year", "2 years", "3 years", "4 years", "10+ years", "5 years"],
    "address_state": ["GA", "IL", "HI", "CA", "MI", "FL"],
    "home_ownership": ["Own", "Rent", "Mortgage", "Rent", "Own", "Mortgage"],
    "loan_status": ["Current", "Fully Paid", "Current", "Charged Off", "Fully Paid", "Charged Off"]
  })
  one_hot_cols = ["address_state", "home_ownership"]

  ## Act
  df = one_hot_encoding(test_data, one_hot_cols)

  ## Assert
  expected_columns = [
    "address_state_FL",
    "address_state_GA",
    "address_state_HI",
    "address_state_IL",
    "address_state_MI",
    "home_ownership_Own",
    "home_ownership_Rent"
  ]

  ## Assert if the original columns have been removed
  assert "address_state" not in df.columns
  assert "home_ownership" not in df.columns

  ## Check that all expected columns are in the df
  assert set(expected_columns).issubset(df.columns)

  ## Check the value of one columns
  expected_address_state = [0, 1, 0, 0, 0, 0]
  assert df["address_state_IL"].tolist() == expected_address_state

## We test the binary encoder
def test_binary_encoding_term():
  test_data = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6],
    "grade": ["A", "A", "C", "D", "E", "B"],
    "application_type": ["test01", "test02", "test03", "test04", "test05", "test06"],
    "emp_title": ["title_test01", "title_test02", "title_test03", "title_test04", "title_test05", "title_test06"],
    "sub_grade": ["A1", "D2", "C3", "B5", "E1", "F2"],
    "emp_length": ["< 1 year", "2 years", "3 years", "4 years", "10+ years", "5 years"],
    "address_state": ["GA", "IL", "HI", "CA", "MI", "FL"],
    "home_ownership": ["Own", "Rent", "Mortgage", "Rent", "Own", "Mortgage"],
    "term": ["36 months", "60 months", "60 months", "60 months", "36 months", "36 months"],
    "loan_status": ["Current", "Fully Paid", "Current", "Charged Off", "Fully Paid", "Charged Off"]
  })

  ## Act
  ## encode term
  df = binary_encoding(test_data, "term")

  ## Assert
  expected_values = [0, 1, 1, 1, 0, 0]
  assert df["term"].tolist() == expected_values

## We test the binary encoder
def test_binary_encoding_term():
  test_data = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6],
    "grade": ["A", "A", "C", "D", "E", "B"],
    "application_type": ["test01", "test02", "test03", "test04", "test05", "test06"],
    "emp_title": ["title_test01", "title_test02", "title_test03", "title_test04", "title_test05", "title_test06"],
    "sub_grade": ["A1", "D2", "C3", "B5", "E1", "F2"],
    "emp_length": ["< 1 year", "2 years", "3 years", "4 years", "10+ years", "5 years"],
    "address_state": ["GA", "IL", "HI", "CA", "MI", "FL"],
    "home_ownership": ["Own", "Rent", "Mortgage", "Rent", "Own", "Mortgage"],
    "term": ["36 months", "60 months", "60 months", "60 months", "36 months", "36 months"],
    "loan_status": ["Current", "Fully Paid", "Current", "Charged Off", "Fully Paid", "Charged Off"]
  })

  ## Act
  df = remove_current(test_data)
  ## encode term
  df = binary_encoding(df, "loan_status")

  ## Assert
  expected_values = [1, 0, 1, 0]
  assert df["loan_status"].tolist() == expected_values
  assert "Current" not in df["loan_status"].values