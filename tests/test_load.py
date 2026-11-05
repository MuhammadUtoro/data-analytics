## Test module for load module
import pandas as pd
import os
from pathlib import Path
from src.load import save_csv

## Test the save_csv()
def test_save_csv(tmp_path):
  filepath = tmp_path
  filename = "test.csv"
  ## Arrange
  test_data = pd.DataFrame({
    "id": [1, 2, 3, 3],
    "address_state": ["IL", "CA", "GA", "FL"],
    "emp_length": ["< 1 year", "1 year", "2 years", "3 years"]
  })

  ## Act
  saved_file = save_csv(test_data, filepath, filename)

  ## Assert
  assert os.path.isfile(saved_file)

## Test the makedirs - if it really creates the directory
def test_save_csv_makedirs(tmp_path):
  ## Arrange
  filepath = tmp_path / "output"
  filename = "test.csv"
  ## Arrange
  test_data = pd.DataFrame({
    "id": [1, 2, 3, 3],
    "address_state": ["IL", "CA", "GA", "FL"],
    "emp_length": ["< 1 year", "1 year", "2 years", "3 years"]
  })

  ## Act
  saved_file = save_csv(test_data, filepath, filename)

  ## Assert
  assert os.path.isdir(filepath)
  assert os.path.isfile(saved_file)