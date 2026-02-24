import pandas as pd
import sqlite3 as sq
import matplotlib.pyplot as plt
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

#task 4
BLOCKS = ['0-4', '4-8', '8-12', '12-16', '16-20', '20-24']

def assign_block(hour):
    if hour < 4:
        return '0-4'
    elif hour < 8:
        return '4-8'
    elif hour < 12:
        return '8-12'
    elif hour < 16:
        return '12-16'
    elif hour < 20:
        return '16-20'
    else:
        return '20-24'

def compute_block_averages(conn):
    #steps
    df_steps = pd.read_sql_query("SELECT Id, ActivityHour, StepTotal FROM hourly_steps", conn)
    df_steps['ActivityHour'] = pd.to_datetime(df_steps['ActivityHour'])
    df_steps['block'] = df_steps['ActivityHour'].dt.hour.apply(assign_block)
    df_steps['date'] = df_steps['ActivityHour'].dt.date
    steps_per_block = df_steps.groupby(['Id', 'date', 'block'])['StepTotal'].sum().reset_index()
    avg_steps = steps_per_block.groupby('block')['StepTotal'].mean().reindex(BLOCKS)

    #calories
    df_cals = pd.read_sql_query("SELECT Id, ActivityHour, Calories FROM hourly_calories", conn)
    df_cals['ActivityHour'] = pd.to_datetime(df_cals['ActivityHour'])
    df_cals['block'] = df_cals['ActivityHour'].dt.hour.apply(assign_block)
    df_cals['date'] = df_cals['ActivityHour'].dt.date
    cals_per_block = df_cals.groupby(['Id', 'date', 'block'])['Calories'].sum().reset_index()
    avg_cals = cals_per_block.groupby('block')['Calories'].mean().reindex(BLOCKS)

    #sleep
    df_min_sleep = pd.read_sql_query("SELECT Id, date, logId FROM minute_sleep", conn)
    df_min_sleep['datetime'] = pd.to_datetime(df_min_sleep['date'])
    df_min_sleep['block'] = df_min_sleep['datetime'].dt.hour.apply(assign_block)
    sleep_per_block = df_min_sleep.groupby(['Id', 'logId', 'block']).size().reset_index(name='sleep_minutes')
    avg_sleep = sleep_per_block.groupby('block')['sleep_minutes'].mean().reindex(BLOCKS)

    return avg_steps, avg_cals, avg_sleep


def plot_block_averages(conn):
    avg_steps, avg_cals, avg_sleep = compute_block_averages(conn)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(BLOCKS, avg_steps, color='steelblue')
    axes[0].set_title('Average Steps per 4-Hour Block')
    axes[0].set_xlabel('Time Block')
    axes[0].set_ylabel('Average Steps')

    axes[1].bar(BLOCKS, avg_cals, color='tomato')
    axes[1].set_title('Average Calories Burnt per 4-Hour Block')
    axes[1].set_xlabel('Time Block')
    axes[1].set_ylabel('Average Calories')

    axes[2].bar(BLOCKS, avg_sleep, color='mediumseagreen')
    axes[2].set_title('Average Sleep Minutes per 4-Hour Block')
    axes[2].set_xlabel('Time Block')
    axes[2].set_ylabel('Average Sleep Minutes')

    plt.tight_layout()
    return fig


# Task 5: Heart rate and exercise intensity visualization

def plot_individual(conn, individual_id):
    df_hr = pd.read_sql_query(
        "SELECT Time, Value FROM heart_rate WHERE CAST(Id AS INTEGER) = ?",
        conn, params=(individual_id,)
    )
    df_hr['Time'] = pd.to_datetime(df_hr['Time'])
    df_hr = df_hr.sort_values('Time')

    df_intensity = pd.read_sql_query(
        "SELECT ActivityHour, TotalIntensity FROM hourly_intensity WHERE CAST(Id AS INTEGER) = ?",
        conn, params=(individual_id,)
    )
    df_intensity['ActivityHour'] = pd.to_datetime(df_intensity['ActivityHour'])
    df_intensity = df_intensity.sort_values('ActivityHour')

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(df_hr['Time'], df_hr['Value'], linewidth=0.6, color='crimson')
    axes[0].set_title(f'Heart Rate Over Time (Id: {individual_id})')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Heart Rate (bpm)')

    axes[1].bar(df_intensity['ActivityHour'], df_intensity['TotalIntensity'],
                width=1/24, color='steelblue')
    axes[1].set_title(f'Total Exercise Intensity (Id: {individual_id})')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Total Intensity')

    plt.tight_layout()
    return fig
