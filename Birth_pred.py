import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
# --- 1. LOAD DATA ---

try:
    births = pd.read_csv("C:/Users/aarja_11ews5y/Downloads/births.csv")
except FileNotFoundError:
    print(f"Error: Could not find 'births.csv' at C:/Users/aarja_11ews5y/Downloads/births.csv")
    exit()

# --- 2. DATA CLEANING & PREPARATION ---
births['day'].fillna(0, inplace=True)
births['day'] = births['day'].astype(int)

# Annual Totals (For the Regression Graph)
yearly_data = births.groupby('year')['births'].sum().reset_index()

# Monthly Totals (For the Prediction Model)
monthly_data = births.groupby(['year', 'month'])['births'].sum().reset_index()

# --- 3. TRAIN MODELS ---
# Model 1: Overall Trend (Yearly)
X_year = yearly_data[['year']]
y_year = yearly_data['births']
model_trend = LinearRegression()
model_trend.fit(X_year, y_year)

# Model 2: Monthly Details (Year + Month)
X_month = monthly_data[['year', 'month']]
y_month = monthly_data['births']
model_month = LinearRegression()
model_month.fit(X_month, y_month)

# --- 4. USER INPUT ---
print("-" * 30)
try:
    user_input = input("Enter the year you want to predict: ")
    target_year = int(user_input)
except ValueError:
    print("Invalid input! Defaulting to 2025.")
    target_year = 2025

# --- 5. CALCULATE PREDICTIONS ---
# Monthly Forecast
future_months = pd.DataFrame({
    'year': [target_year] * 12,
    'month': list(range(1, 13))
})
future_months['predicted_births'] = model_month.predict(future_months).astype(int)

# Extended Trend Line (Future)
# Create a range of years from the start of data up to the user's target year
min_year = yearly_data['year'].min()
max_year = max(yearly_data['year'].max(), target_year)
trend_range = pd.DataFrame({'year': np.arange(min_year, max_year + 1)})
trend_values = model_trend.predict(trend_range)

print(f"\n--- Forecast for {target_year} ---")
print(future_months[['month', 'predicted_births']].to_string(index=False))
print("-" * 30)

# --- 6. VISUALIZATION ---
sns.set_theme(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
plt.subplots_adjust(hspace=0.4)

# --- CHART 1: Overall Trend
# 1. Plot Historical Dots
sns.scatterplot(x='year', y='births', data=yearly_data, ax=ax1, 
                color='#1f77b4', s=60, label='Historical Data', alpha=0.7)

# 2. Plot the Regression Line (Extended)
ax1.plot(trend_range['year'], trend_values, color='#d62728', linewidth=2, label='Trend Line (Extrapolated)')

# 3. Mark the Future Prediction on the line
future_val = model_trend.predict([[target_year]])[0]
ax1.scatter([target_year], [future_val], color='green', s=100, zorder=5, label=f'Prediction {target_year}')

ax1.set_title(f"Historical Trend with Future Projection to {target_year}", fontsize=14)
ax1.set_xlabel("Year")
ax1.set_ylabel("Total Annual Births")
ax1.legend()

# --- CHART 2: Monthly Prediction ---
ax2.plot(future_months['month'], future_months['predicted_births'], 
         marker='o', linestyle='-', color='teal', linewidth=2)

ax2.set_title(f"Detailed Prediction for {target_year}", fontsize=14)
ax2.set_xlabel("Month")
ax2.set_ylabel("Predicted Births")
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

print("Displaying graphs...")
plt.show()