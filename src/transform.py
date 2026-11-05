## Transform module - In this module the goal is to prepare the data for Machine Learning testing
import pandas as pd

## At the moment, we still have the "current" as the payment status, which is not going to be used as a target
def remove_current(df):
  df = df.loc[df["loan_status"] != "Current"]
  return df

grade_map = {
  "G": 0,
  "F": 1,
  "E": 2,
  "D": 3,
  "C": 4,
  "B": 5,
  "A": 6
}
sub_grade_map = {
  "G5": 0,
  "G4": 1,
  "G3": 2,
  "G2": 3,
  "G1": 4,
  "F5": 5,
  "F4": 6,
  "F3": 7,
  "F2": 8,
  "F1": 9,
  "E5": 10,
  "E4": 11,
  "E3": 12,
  "E2": 13,
  "E1": 14,
  "D5": 15,
  "D4": 16,
  "D3": 17,
  "D2": 18,
  "D1": 19, "C5": 20, "C4": 21, "C3": 22, "C2": 23, "C1": 24,
  "B5": 25, "B4": 26, "B3": 27, "B2": 28, "B1":29, "A5": 30,
  "A4": 31, "A3": 32, "A2": 33, "A1": 34
}

emp_length_map = {
  "< 1 year": 0,
  "1 year": 1,
  "2 years": 2,
  "3 years": 3,
  "4 years": 4,
  "5 years": 5,
  "6 years": 6,
  "7 years": 7,
  "8 years": 8,
  "9 years": 9,
  "10+ years": 10
}
## We apply ordinal encoding for "grade", "sub_grade" and "emp_length" so it can be used for training
def map_ordinal(df, col_name: str) -> pd.DataFrame:

  if (col_name == "grade"):
    df[col_name] = df[col_name].map(grade_map)
  elif (col_name == "sub_grade"):
    df[col_name] = df[col_name].map(sub_grade_map)
  else:
    df[col_name] = df[col_name].map(emp_length_map)

  return df

## Now we remove the features which are unrelated or indirectly reveals the targets
## The columns are: 'id', 'application_type', 'emp_title', 'last_credit_pull_date', 'last_payment_date', 'issue_date', 'next_payment_date', 'total_payment', 'member_id
def remove_unrelated_columns(df, columns: list[str]) -> pd.DataFrame:
  df = df.drop(columns=columns)
  return df

## We do one-hot encoding for categorical columns
## Columns are: address_state, home_ownership, verification_status, purpose
def one_hot_encoding(df, columns: list[str]) -> pd.DataFrame:
  df = pd.get_dummies(
    df,
    columns=columns,
    drop_first=True,
  )
  return df

## We do binary encoding for term and loan_status (target)
term_map = {
  "36": 0,
  "60": 1
}
status_map = {
  "Charged Off": 0,
  "Fully Paid": 1
}
def binary_encoding(df, col_name: str) -> pd.DataFrame:
  if (col_name == "term"):
    df[col_name] = df[col_name].astype(str).str.extract(r"(\d+)", expand=False).map(term_map)
  else:
    df[col_name] = df[col_name].map(status_map)

  return df