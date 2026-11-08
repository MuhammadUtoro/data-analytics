# %%
## Bank Loan Analysis
## Import pacakges/libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %%
## Load and Observe RAW Data
df = pd.read_excel("../data/raw/financial_loan.xlsx")
df.info()
# %%
## Quick statistical overview
df.describe()
# %%
## First analysis task: Calculate the total number of loan applications.
## Additionally, calculate also the month-to-date loan applications
issue_date = df["issue_date"]
total_applications = issue_date.count()
print(f"total applications: {total_applications}")
# %%
## Calculate the MTD (month to date)
start_date = issue_date.min()
latest_date = issue_date.max()
latest_date_year = latest_date.year
latest_date_month = latest_date.month
mtd_data = df[(df["issue_date"].dt.year == latest_date_year) & (df["issue_date"].dt.month == latest_date_month)]
print(f"total month-to-date applications: {mtd_data}")
# %%
## Second task: Calculate the total amount funded
## It is the total amount funded by the bank
## We split the data into all time and month-to-date
df_all_time = df.copy()
df_mtd = mtd_data
# %%
## Do the aggregation
amount_funded_all_time = df_all_time["loan_amount"].sum()
print(f"Total amount received (all-time): {amount_funded_all_time/1000000:.2f}M")
amount_funded_mtd = df_mtd["loan_amount"].sum()
print(f"total amount per month: {amount_funded_mtd/1000000:.2f}M")
# %%
## Third task: Calculate the total amount of received
## It is the total amount the applicants paid plus interest
amount_received_all_time = df_all_time["total_payment"].sum()/1000000
print(f"total amount received: {amount_received_all_time:.2f}M")
amount_received_mtd = df_mtd["total_payment"].sum()/1000000
print(f"total amount received monthly: {amount_received_mtd:.2f}")
# %%
## Fourth task: Find the average interest rate
## We can actually find it from the quick statistical overview
avg_interest_rate = df_all_time.describe()["int_rate"]["mean"]*100
print(f"average interest rate: {avg_interest_rate:.2f}%")
# %%
## Fifth task: Calculate the Debt to Income Ratio
## It is used as a parameter to observe the financial health of the applicants
dti_ratio = df_all_time.describe()["dti"]["mean"]*100
print(f"Debt to Income ratio: {dti_ratio:.2f}")
# %%
## Sixth task: We split the data into bad and good loans
## Bad Loans: loan_status is charged off
## Good Loans: loan_status is fully paid/current
df_good_loans = df_all_time[df_all_time["loan_status"].isin(["Fully Paid", "Current"])]
df_bad_loans = df_all_time[df_all_time["loan_status"].isin(["Charged Off"])]
# %%
## Calculate the percentage
good_loans_pct = (df_good_loans.count()["id"]/df_all_time.count()["id"])*100
bad_loans_pct = (df_bad_loans.count()["id"]/df_all_time.count()["id"])*100
print(
  f"Good loans percentage: {good_loans_pct:.2f}%\n"
  f"Bad loans percentage: {bad_loans_pct:.2f}%"
)
# %%
## Seventh Task: Calculate total money received and spent from each bad and good loan
