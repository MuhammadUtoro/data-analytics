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
## Create function to calculate the parameters
def loanMetrics(df: pd.DataFrame, cat: str):
  total_spent_in_mil = df["loan_amount"].sum()/1000000
  total_received_in_mil = df["total_payment"].sum()/1000000
  profit = total_received_in_mil - total_spent_in_mil
  print(
    f"Total spent ({cat}) loan: ${total_spent_in_mil:.2f}M\nTotal received ({cat}) loan: ${total_received_in_mil:.2f}M\nTotal profit ({cat}): ${profit:.2f}M"
  )
# %%
good_loans_metric = loanMetrics(df_good_loans, "good")
bad_loans_metric = loanMetrics(df_bad_loans, "bad")
total_metric = loanMetrics(df_all_time, "total")
# %%
## Visualisations Part
## First task: Monthly trend graph
import matplotlib.dates as mdates
def plotRunningTotal(df):
  monthly_traffic = df.sort_values("issue_date").groupby(pd.Grouper(key="issue_date", freq="ME"))[["loan_amount", "total_payment"]].sum().div(1000000).reset_index()
  plt.figure(figsize=(10, 8))
  plt.plot(
    monthly_traffic["issue_date"],
    monthly_traffic["loan_amount"],
    color="red",
    label="amount spent",
    linewidth=2
  )
  plt.plot(
    monthly_traffic["issue_date"],
    monthly_traffic["total_payment"],
    color="blue",
    label="amount received",
    linewidth=2
  )
  ## Add mark above the line
  for i, row in monthly_traffic.iterrows():
    plt.text(
      row["issue_date"], row["loan_amount"], f"{row["loan_amount"]:.2f}",
      ha="center", va="bottom", color="red"
    )
  for i, row in monthly_traffic.iterrows():
    plt.text(
      row["issue_date"], row["total_payment"], f"{row["total_payment"]:.2f}",
      ha="center", va="bottom", color="blue"
    )
  plt.title('Monthly Money Spent')
  plt.xlabel('Month')
  plt.ylabel('Money Spent in Millions')
  plt.xticks(rotation=45)
  plt.grid(True)
  plt.legend()
  plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
  plt.show()
# %%
plotRunningTotal(df_all_time)
# %%
## Monthly trend analysis
trend_count = df_all_time.groupby(df_all_time['issue_date'].dt.strftime('%b %Y'))['id'].count().sort_index()

# %%
fig, ax = plt.subplots()

line = ax.plot(
    trend_count.index,
    trend_count.values,
    marker='o'
)
for x, y in zip(trend_count.index, trend_count.values):
    ax.text(x, y, str(y))
ax.set_title('Monthly Loan Trends')
ax.set_ylabel('Loan count')
plt.show()

# %%
## Third task: Regional analysis
# First, we calculate the amount spent (loan_amount per state)
total_running_per_state = df_all_time.groupby('address_state')[['loan_amount', 'total_payment']].sum().div(1000000).reset_index()
top_5_state = total_running_per_state.sort_values(['loan_amount', 'total_payment'], ascending=False).head(5)
top_5_state = top_5_state.rename(
    columns={
        'address_state': 'State',
        'loan_amount': 'Money Spent',
        'total_payment': 'Money Received'
    }
)
print(top_5_state)
# %%
# Then we plot using bar char for comparison
x = np.arange(len(top_5_state['State']))
width = 0.4
plt.figure(figsize=(8,5))
# Add label on the top of the bar
bars1 = plt.bar(x - width/2, top_5_state['Money Spent'], width, label='Money Spent')
bars2 = plt.bar(x + width/2, top_5_state['Money Received'], width, label='Money Received' )
plt.bar_label(bars1, fmt='$%.1fM')
plt.bar_label(bars2, fmt='$%.1fM')
plt.xticks(x, top_5_state['State'], rotation=45)
plt.legend()
plt.title('Top 5 States with Highest Loan Amount')
plt.ylabel('Total Amount (Millions)')
plt.show()
# %%
## Extended task: Comparing top 5 states with the whole country
states = top_5_state['State'].tolist()
values_spent = top_5_state['Money Spent'].tolist()
values_received = top_5_state['Money Received'].tolist()

others_spent = (
    total_running_per_state['loan_amount'].sum() - sum(values_spent)
)
others_received = (
    total_running_per_state['total_payment'].sum() - sum(values_received)
)
states.append('Others')
values_spent.append(others_spent)
values_received.append(others_received)

# Plot
fig, axes = plt.subplots(
    1,2,figsize=(12, 6)
)
axes[0].pie(
    values_spent,
    labels=states,
    autopct='%.1f%%'
)
axes[0].set_title('Percentage of Loan Amount')

axes[1].pie(
    values_received,
    labels=states,
    autopct='%.1f%%'
)
axes[1].set_title('Percentage of Total Payment')
plt.show()
# %%
## Extended task: Loan purpose


loan_purpose_count = df_all_time.groupby('purpose', sort=False).count()['id'].sort_values(ascending=False).reset_index(name='purpose_count')
loan_purpose_count

fig, ax = plt.subplots()
bars = ax.barh(
    loan_purpose_count['purpose'],
    loan_purpose_count['purpose_count']
)
ax.bar_label(bars, fmt='%.0f')
ax.set_title('Loan Purpose Breakdown')
plt.show()

# %%
