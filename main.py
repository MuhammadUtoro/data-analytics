## Orchestrate the ETL Pipeline
from src.extract import *
from src.transform import *
from src.load import save_csv

def main():
  filepath = "data/raw"
  filename = "financial_loan.xlsx"
  saved_filepath = "data/transformed"
  saved_filename = "transformed_data.csv"
  unrelated_cols = ["id", "application_type", "emp_title",
                    "last_credit_pull_date", "last_payment_date", "issue_date",
                    "next_payment_date", "total_payment", "member_id"
                    ]

  one_hot_cols = [
    "address_state",
    "home_ownership",
    "verification_status",
    "purpose"
  ]

  ## Extract
  df_raw = extract_excel(filepath, filename)

  ## Transform
  df_remove_current = remove_current(df_raw)
  df_grade_mapped = map_ordinal(df_remove_current, "grade")
  df_subgrade_mapped = map_ordinal(df_grade_mapped, "sub_grade")
  df_emp_length_mapped = map_ordinal(df_subgrade_mapped, "emp_length")
  df_cleaned = remove_unrelated_columns(df_emp_length_mapped, unrelated_cols)
  df_cleaned = one_hot_encoding(df_cleaned, one_hot_cols)
  df_cleaned = binary_encoding(df_cleaned, "term")
  df_cleaned = binary_encoding(df_cleaned, "loan_status")

  ## Load
  saved_df = save_csv(df_cleaned, saved_filepath, saved_filename)

  print(df_cleaned.head(5))

if __name__ == "__main__":
  main()


