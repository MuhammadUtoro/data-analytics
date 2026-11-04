## Extract module to load data
import pandas as pd

def extract_excel(filename):
  df = pd.read_excel(filename)
  print(df.head(5))
  return df
