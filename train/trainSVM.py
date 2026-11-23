# %%
## We use Support Vector Machine Algorithm (SVM)
import pandas as pd
import numpy as np
from preprocessing import *
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# %%
## Load the transformed data
df = pd.read_csv("../data/transformed/transformed_data.csv")
df.info()
# %%
## Split target and features - using preprocessing
X, y = xy_split(df, "loan_status")
print(
  f"X-shape: {X.shape}"
  f"y-shape: {y.shape}"
)
# %%
## split into train and test
X_train, X_test, y_train, y_test = traintest_split(X, y)
# %%
## Create base pipline for SVM
base_pipeline = Pipeline([
  ("scaler", StandardScaler()),
  ("linSVC_clf", LinearSVC(
    C=1.0,
    penalty="l2",
    loss="squared_hinge",
    class_weight="balanced",
    max_iter=1000,
    random_state=42
  ))
])
# %%
## Fit the model
base_pipeline.fit(X_train, y_train)
# %%
## Check score
y_pred_base = base_pipeline.predict(X_test)
base_pipeline.score(X_test, y_test)
# %%
## Evaluate the model
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
confuse_mat_base = confusion_matrix(y_test, y_pred_base)
class_report_base = classification_report(y_test, y_pred_base)
# %%
## Print the result
print(confuse_mat_base)
print("-------------------------")
print(class_report_base)
# %%
