# %% We use logistic regression
import pandas as pd
import numpy as np
from preprocessing import *
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# %%
## We load the transformed CSV data
df = pd.read_csv("../data/transformed/transformed_data.csv")
df.info()
# %%
## Split target and features
X, y = xy_split(df, "loan_status")
print(
  f"X-shape: {X.shape}"
  f"y-shape: {y.shape}"
)
# %%
## Train test split
X_train, X_test, y_train, y_test = traintest_split(X, y)
# %%
## Create Pipeline for logistic regression
base_pipeline = Pipeline([
  ("scaler", StandardScaler()),
  ("log_reg", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
])
# %%
## Fit the model
base_pipeline.fit(X_train, y_train)
# %%
## Predict and find the score
y_pred = base_pipeline.predict(X_test)
base_pipeline.score(X_test, y_test)
# %%
## Evaluate the model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# %%
