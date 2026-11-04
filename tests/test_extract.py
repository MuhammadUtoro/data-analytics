## Test file for extract module
import pandas as pd
from src.extract import extract_excel

## Recall - Unit test. Independent of the actual data
def test_extract_excel(tmp_path):
  filepath = tmp_path
  filename = "test.xlsx"
  ## Arrange
  test_data = pd.DataFrame({
    "id": ["1456", "2223", "4322", "1111"],
    "address_state": ["GA", "IL", "TX", "CA"],
    "grade": ["A", "B", "C", "C"]
  })

  ## Save test data to excel
  test_data.to_excel(
    tmp_path / filename,
    index=False
  )

  ## Act
  df = extract_excel(filepath / filename)

  ## Assert
  assert isinstance(df, pd.DataFrame)
  assert not df.empty

## Test if the dataframe is the exact shape
def test_extract_shape(tmp_path):
  ## Arrange
  filepath = tmp_path
  filename = "test.xlsx"
  ## Arrange
  test_data = pd.DataFrame({
    "id": ["1456", "2223", "4322", "1111"],
    "address_state": ["GA", "IL", "TX", "CA"],
    "grade": ["A", "B", "C", "C"]
  })

  ## Export to excel
  test_data.to_excel(
    filepath / filename,
    index=False
  )

  ## Act
  df = extract_excel(filepath / filename)

  ## Assert
  assert df.shape == test_data.shape


