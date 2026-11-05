## Extract module to load data
import pandas as pd
from pathlib import Path

def extract_excel(filepath, filename):
  file = Path(filepath, filename)
  df = pd.read_excel(file)
  print(df.head(5))
  return df

filepath = "data/raw"
filename = "financial_loan.xlsx"
file = Path(filepath, filename)

df = extract_excel(filepath, filename)
