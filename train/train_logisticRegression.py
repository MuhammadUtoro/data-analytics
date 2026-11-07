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
y_pred_base = base_pipeline.predict(X_test)
base_pipeline.score(X_test, y_test)
# %%
## Evaluate the model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
confuse_mat_base = confusion_matrix(y_test, y_pred_base)
class_report_base = classification_report(y_test, y_pred_base, output_dict=True)
# %%
## Now we do Hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV
grid_search = [
  {"log_reg__class_weight": ["balanced"], "log_reg__solver": ["lbfgs"], "log_reg__C": [10, 100], "log_reg__max_iter": [100, 1000]},
]

scoring = {
  "recall": "recall",
  "precision": "precision",
  "f1": "f1",
  "accuracy": "accuracy"
}

gridCV = GridSearchCV(
  estimator=base_pipeline,
  param_grid=grid_search,
  scoring=scoring,
  refit="recall",
  cv=3,
  return_train_score=True,
  n_jobs=-1
)

# %%
## Fit the model
gridCV.fit(X_train, y_train)
# %%
## Find the best estimator
gridCV.best_estimator_
# %%
print(gridCV.best_params_)
print(gridCV.best_score_)
# %%
## Use the best model to predict
y_pred_best = gridCV.best_estimator_.predict(X_test)
# %%
confuse_mat_best = confusion_matrix(y_test, y_pred_best)
class_report_best = classification_report(y_test, y_pred_best, output_dict=True)
# %%
## Visualise the comparison between parameters
## Best model
class_report_matrix_best = pd.DataFrame(class_report_best).T
class_report_matrix_best = class_report_matrix_best.loc[["0","1"], ["precision", "recall", "f1-score"]]
class_report_matrix_best
# %%
## Base model
class_report_matrix_base = pd.DataFrame(class_report_base).T
class_report_matrix_base = class_report_matrix_base.loc[["0", "1"], ["precision", "recall", "f1-score"]]
class_report_matrix_base
# %%
## Compare
print("Baseline Model Classification Report")
print("-------------------------------------")
print(class_report_matrix_base)
# %%
print("Best Model (after tuning) Classification Report")
print("-------------------------------------")
print(class_report_matrix_best)
# %%
