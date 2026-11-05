## Module for load the data
import os
import pandas as pd

def save_csv(df, filepath, filename):
  os.makedirs(filepath, exist_ok=True)

  file = os.path.join(filepath, filename)

  df.to_csv(file, index=False)

  print(f"file is successfully saved to: {file}")
  return file