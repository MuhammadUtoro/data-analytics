# %%
## Now we try Random Forest
import pandas as pd
import numpy as np
from preprocessing import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
# %%
## Load the transformed data
df = pd.read_csv("../data/transformed/transformed_data.csv")
df.info()
# %%
## Split into features and target
X, y = xy_split(df, "loan_status")
# %%
## Split into training and test
X_train, X_test, y_train, y_test = traintest_split(X, y)
# %%
## We create the pipeline for the model
## We don't use scaler, since random forest isn't sensitive to scaling
base_pipeline = Pipeline([
  ("rf_clf", RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
  ))
])
# %%
## Fit the model
base_pipeline.fit(X_train, y_train)
# %%
## Predict and observe the score
y_pred = base_pipeline.predict(X_test)
# %%
base_pipeline.score(X_test, y_test)
# %%
## Evaluate the other parameters
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix_base = confusion_matrix(y_test, y_pred)
class_report_base = classification_report(y_test, y_pred, output_dict=True)
print(confusion_matrix_base)
print(class_report_base)
# %%
## We do hyperparameter tuning using GridSearchCV and find the best estimator
from sklearn.model_selection import GridSearchCV
grid_search = [
  {"rf_clf__n_estimators": [100, 300, 500],
   "rf_clf__max_depth": [20, 30, None],
   "rf_clf__min_samples_split": [5, 10, 20],
   "rf_clf__min_samples_leaf": [4, 8, 10]}
]

scoring = {
  "recall": "recall",
  "precision": "precision",
  "f1": "f1",
  "accuracy": "accuracy"
}

# %%
gridSearchCV = GridSearchCV(
  estimator=base_pipeline,
  param_grid=grid_search,
  scoring=scoring,
  refit="recall",
  cv=3,
  return_train_score=True
)
# %%
## Fit the model
gridSearchCV.fit(X_train, y_train)
# %%
## Now we fit the best model
gridSearchCV.best_estimator_
# %%
best_param = gridSearchCV.best_params_
best_score = gridSearchCV.best_score_
print(best_param, best_score)
# %%
## Use the best parameters to predict
y_pred_best = gridSearchCV.predict(X_test)
gridSearchCV.score(X_test, y_test)
# %%
confusion_matrix_best = confusion_matrix(y_test, y_pred)
class_report_best = classification_report(y_test, y_pred)
print(confusion_matrix_best)
print(class_report_best)
# %%
