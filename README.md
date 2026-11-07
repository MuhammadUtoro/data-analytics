# Bank Loan Analysis 📊

This project explores bank loan data using Python, focusing on loan distribution, repayment behavior, and key financial patterns across different states, purposes, and time periods.

The project has since been extended to include a Machine Learning component, with the objective of predicting loan repayment outcomes based on historical financial data.

The dataset and initial problem statements are provided by [Data Tutorials](https://www.youtube.com/@datatutorials1). Thank you very much for providing the dataset and learning resources!

## 📌 Project Goals

- Analyze loan amounts and total payments.
- Identify top states by loan volume.
- Understand loan purpose distribution.
- Compare money spent vs. money received.
- Explore monthly loan trends.
- Develop a Machine Learning model to predict loan repayment outcomes (Charged Off or Fully Paid).
- Evaluate and compare different classification algorithms.

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 📊 Key Analyses

- Grouping and aggregation of loan data.
- Time series analysis (monthly trends).
- State-wise loan comparison.
- Loan purpose breakdown.
- Data visualization using bar charts, pie charts, and line plots.

## 📈 Visualizations

The project includes:

- Grouped bar charts (comparison between spent and received amounts).
- Pie charts (percentage distribution).
- Horizontal bar charts (loan purpose breakdown).
- Line plots (monthly trends).
- Confusion matrices and classification reports for Machine Learning evaluation.

---

# 🤖 Machine Learning – Loan Status Prediction

## 🎯 Objective

The Machine Learning component aims to predict the outcome of a loan application based on the available financial and borrower-related features.

The prediction task is formulated as a binary classification problem with two possible outcomes:

- **Charged Off (Class 0):** The borrower failed to fully repay the loan.
- **Fully Paid (Class 1):** The borrower successfully repaid the loan.

The objective is to identify patterns in historical loan data that can help distinguish between these two outcomes.

## ⚙️ Data Preprocessing and Workflow

Before training the classification models, several preprocessing steps were performed:

1. **Data Cleaning:** Removed records with the `Current` loan status to ensure the classification task contains only two target classes.
2. **Encoding:** Converted categorical variables into numerical representations and encoded the target variable (`loan_status`) into binary values.
3. **Feature and Target Separation:** Separated the independent variables (features) from the dependent variable (target).
4. **Train-Test Split:** Divided the dataset into training and testing subsets.
5. **Baseline Modeling:** Established baseline classification models to evaluate initial performance.
6. **Model Evaluation:** Assessed model performance using precision, recall, F1-score, accuracy, and confusion matrices.
7. **Hyperparameter Tuning:** Applied `GridSearchCV` to explore different hyperparameter combinations and improve model performance.
8. **Model Comparison:** Compared baseline and tuned models to identify potential improvements.

## 🧠 Classification Algorithms

Two classification algorithms were initially explored:

- Logistic Regression
- Random Forest Classifier

Both algorithms were evaluated using multiple classification metrics, with particular attention given to precision and recall due to the imbalanced distribution of the target classes.

### 1. Logistic Regression

The baseline Logistic Regression model produced the following results:

| Metric | Charged Off (Class 0) | Fully Paid (Class 1) |
|---|---:|---:|
| Precision | 23% | 91% |
| Recall | 63% | 64% |

**Interpretation:**

The model demonstrated relatively similar recall scores for both classes, identifying approximately 63% of actual Charged Off loans and 64% of actual Fully Paid loans.

However, the precision scores revealed a considerable difference between the two classes.

For Charged Off loans, the precision was approximately 23%. This means that only 23% of the loans predicted as Charged Off were actually Charged Off.

In contrast, the precision for Fully Paid loans was approximately 91%, indicating that the model was considerably more reliable when predicting this class.

After hyperparameter tuning using `GridSearchCV`, the best estimator was identified and refitted using the training dataset. However, the resulting performance was almost identical to the baseline model, suggesting that the explored hyperparameter combinations did not provide a significant improvement.

### 2. Random Forest Classifier

The baseline Random Forest Classifier produced the following results:

| Metric | Charged Off (Class 0) | Fully Paid (Class 1) |
|---|---:|---:|
| Precision | 18.4% | 87% |
| Recall | 34.3% | 94% |

**Interpretation:**

Compared to Logistic Regression, the Random Forest model demonstrated considerably higher recall for Fully Paid loans (94%), meaning that it successfully identified a larger proportion of actual Fully Paid loans.

However, its ability to identify Charged Off loans was considerably weaker, with a recall of only 34.3%.

Furthermore, the precision for Charged Off loans was approximately 18.4%, indicating that a substantial proportion of its Charged Off predictions were incorrect.

After hyperparameter tuning, the best estimator produced almost identical results to the baseline model.

This suggests that the initial hyperparameter search did not result in a meaningful performance improvement.

## 📊 Model Comparison and Evaluation

Based on the initial experiments, the following observations can be made:

- Logistic Regression demonstrated a more balanced recall between the two classes (approximately 63% and 64%).
- Random Forest achieved considerably higher recall for Fully Paid loans (94%), but struggled to identify Charged Off loans (34.3%).
- Both models demonstrated relatively low precision for the Charged Off class.
- Hyperparameter tuning did not produce significant improvements over the baseline models.

Although Random Forest achieved a higher accuracy score (approximately 84%) compared to Logistic Regression (approximately 64%), accuracy alone is insufficient to determine which model performs better.

This is particularly important because the dataset contains an imbalance between the two target classes.

Consequently, greater emphasis is placed on precision, recall, and F1-score to obtain a more comprehensive understanding of model performance.

## 🚧 Future Improvements

The Machine Learning component is still a work in progress. Several areas could be explored to improve the models and evaluation process:

- **F1-score Analysis:** Further investigate the balance between precision and recall, particularly for the Charged Off class.
- **Confusion Matrix Analysis:** Examine false positives and false negatives to better understand classification errors.
- **Class Imbalance Handling:** Experiment with different class-weighting strategies and resampling techniques.
- **Additional Algorithms:** Explore other classification algorithms, such as Decision Trees, Support Vector Machines (SVM), and Gradient Boosting.
- **Hyperparameter Optimization:** Experiment with additional hyperparameters and broader search strategies.
- **Classification Threshold Optimization:** Investigate whether adjusting the default classification threshold can improve recall for the Charged Off class.
- **Feature Importance:** Identify which financial and borrower-related features contribute most to the predictions.

The classification reports and confusion matrices are available in the `figures` directory, alongside the visualizations and interpretations from the exploratory data analysis.

## 📝 Conclusion

This project has evolved from a traditional exploratory data analysis into a broader Machine Learning exercise.

The initial classification experiments demonstrate that different algorithms can produce significantly different results depending on the target class and evaluation metric.

While the models show promising results in identifying Fully Paid loans, predicting Charged Off loans remains a challenge.

Further experimentation with feature engineering, class imbalance strategies, classification thresholds, and additional algorithms will be necessary to improve the predictive performance.

The project serves as an opportunity to explore the complete Machine Learning workflow, from data preprocessing and model development to hyperparameter optimization and model evaluation.
