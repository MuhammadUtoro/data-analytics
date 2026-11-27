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
tn_base, fp_base, fn_base, tp_base = confusion_matrix(y_test, y_pred_base).ravel()
class_report_base = classification_report(y_test, y_pred_base)
# %%
## Print the result
print(np.array([[tn_base, fp_base], [fn_base, tp_base]]))
print("-------------------------")
print(class_report_base)
# %%
## We use hyperparameter tuning
from sklearn.model_selection import GridSearchCV
grid_search = [
  {"linSVC_clf__loss": ["square_hinge", "hinge"], "linSVC_clf__C": [1, 10, 50, 100], "linSVC_clf__max_iter": [100, 1000], "linSVC_clf__class_weight": ["balanced"]}
]
scoring = {
  "recall": "recall",
  "precision": "precision",
  "f1": "precision",
  "accuracy": "accuracy"
}
gridCV = GridSearchCV(
  estimator=base_pipeline,
  param_grid=grid_search,
  scoring=scoring,
  refit="precision",
  cv=3,
  return_train_score=True,
  n_jobs=-1
)
# %%
## Fit the model
gridCV.fit(X_test, y_test)
# %%
## Find the best estimatoe. According to the gridCV, the tuning is:
## loss = hinge, c = 50, max_iter = 100, penalty = l2
print(gridCV.best_params_)
print(gridCV.best_score_)
# %%
## Predict with the best estimator
y_pred_best_svc = gridCV.best_estimator_.predict(X_test)
# %%
## Observe the confusion matrix the classification report
tn_best, fp_best, fn_best, tp_best = confusion_matrix(y_test, y_pred_best_svc).ravel()
class_report_best = classification_report(y_test, y_pred_best_svc)
# %%
print(np.array([
  [tn_best, fp_best],
  [fn_best, tp_best]
]))
print("--------------------------")
print(class_report_best)
# %%
## Now we'll try Non-linear Model SVC
from sklearn.svm import SVC
base_pipeline_svc = Pipeline([
  ("scaler", StandardScaler()),
  ("svc_clf", SVC(
    C=1.0,
    kernel="poly",
    degree=3,
    gamma="scale",
    class_weight="balanced",
    max_iter=1000
  ))
])
# %%
base_pipeline.fit(X_train, y_train)
# %%
## Check score
y_pred_base_svc = base_pipeline.predict(X_test)
base_pipeline.score(X_test, y_test)
# %%
## Classification report and confusion matrix
tn_base_svc, fp_base_svc, fn_base_svc, tp_base_svc = confusion_matrix(
  y_test, y_pred_base_svc
).ravel()
class_report_base_svc = classification_report(y_test, y_pred_base_svc)
# %%
print(np.array([
  [tn_base_svc, fp_base_svc],
  [fn_base_svc, tp_base_svc]
]))
print("--------------------------")
print(class_report_base_svc)
# %%
from sklearn.metrics import recall_score
recall_score(y_test, y_pred_base_svc)
# %%
## Do Hyperparameter tuning
grid_search = [
  {
    "svc_clf__C": [10, 50, 200],
    "svc_clf__kernel": ["poly", "sigmoid", "rbf"],
    "svc_clf__gamma": [1, 10, 50],
    "svc_clf__class_weight": ["balanced"],
    "svc_clf__max_iter": [100, 1000],
    "svc_clf__degree": [2, 3, 5]
  }
]
scoring = {
  "recall": "recall",
  "precision": "precision",
  "f1": "f1",
  "accuracy": "accuracy"
}
gridCV = GridSearchCV(
  estimator=base_pipeline_svc,
  param_grid=grid_search,
  scoring=scoring,
  refit="precision",
  cv=3,
  return_train_score=True,
  n_jobs=-1
)
# %%
gridCV.fit(X_train, y_train)
# %%
print(gridCV.best_params_)
print(gridCV.best_score_)
# %%
y_pred_best = gridCV.best_estimator_.predict(X_test)
# %%
## Observe the confusion matrix the classification report
tn_best, fp_best, fn_best, tp_best = confusion_matrix(y_test, y_pred_best).ravel()
class_report_best = classification_report(y_test, y_pred_best)
# %%
print(np.array([
  [tn_best, fp_best],
  [fn_best, tp_best]
]))
print("--------------------------")
print(class_report_best)
# %%
