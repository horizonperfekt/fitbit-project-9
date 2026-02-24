import pandas as pd
import sqlite3 as sq
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Classification

def classify_ids(df):
    df_id_class = pd.DataFrame(columns=['Id', 'Class'])

    for id in df['Id'].unique():
        count = df.loc[df['Id'] == id, 'Count'].values[0]
        if count <= 10:
            df_id_class = pd.concat([df_id_class, pd.DataFrame({'Id': [id], 'Class': ['Light user']})], ignore_index=True)
        elif count <= 15:
            df_id_class = pd.concat([df_id_class, pd.DataFrame({'Id': [id], 'Class': ['Moderate user']})], ignore_index=True)
        else:
            df_id_class = pd.concat([df_id_class, pd.DataFrame({'Id': [id], 'Class': ['Heavy user']})], ignore_index=True)

    order = ['Light user', 'Moderate user', 'Heavy user']
    df_id_class['Class'] = pd.Categorical(df_id_class['Class'], categories=order, ordered=True)
    df_id_class = df_id_class.sort_values('Class').reset_index(drop=True)

    return df_id_class



df = pd.read_csv("daily_activity.csv")

id_counts = df.groupby('Id').size().reset_index(name='Count')

classified_df = classify_ids(id_counts)


#Database analysis

conn = sq.connect("fitbit_database.db")
cursor = conn.cursor()

df_sleep = pd.read_sql_query("""
    SELECT CAST(Id as INTEGER) as Id, date, logId, COUNT(*) as sleep_duration_minutes
    FROM minute_sleep
    GROUP BY Id, logId
""", conn)

df_active = pd.read_sql_query("""
    SELECT CAST(Id as INTEGER) as Id, ActivityDate as date, 
    VeryActiveMinutes + FairlyActiveMinutes + LightlyActiveMinutes as total_active_minutes
    FROM daily_activity
""", conn)

df_sleep['date'] = pd.to_datetime(df_sleep['date']).dt.date
df_active['date'] = pd.to_datetime(df_active['date']).dt.date


df_merged = pd.merge(df_sleep, df_active, on=['Id', 'date'], how='inner')

print(f"Data inspection: \n {df_merged.head(10)}")

X = df_merged[['total_active_minutes']].values
y = df_merged['sleep_duration_minutes'].values

model = LinearRegression()
model.fit(X, y)

# Results
print("Linear regression: Sleep duration vs Active minutes ===")
print(f"Slope (coefficient): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.2f} minutes")
print(f"R²: {model.score(X, y):.4f}")

plt.figure(figsize=(8,5))
plt.scatter(X, y, alpha=0.5, label='Data points')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression line')
plt.xlabel('Total Active minutes (very + fairly + light)')
plt.ylabel('Sleep duration (minutes)')
plt.title('Sleep vs. Active minutes (all users)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("Per-individual Regressions")
for user_id in df_merged['Id'].unique():
    user_data = df_merged[df_merged['Id'] == user_id]
    # Need at least 2 points to fit a line
    if len(user_data) > 1:
        X_u = user_data[['total_active_minutes']].values
        y_u = user_data['sleep_duration_minutes'].values
        model_u = LinearRegression()
        model_u.fit(X_u, y_u)
        print(f"User {user_id}: slope = {model_u.coef_[0]:.4f}, R² = {model_u.score(X_u, y_u):.4f} (n={len(user_data)})")
    else:
        print(f"User {user_id}: insufficient data (only {len(user_data)} day)")
