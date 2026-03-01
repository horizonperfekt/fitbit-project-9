import os
import numpy as np
import pandas as pd
import sqlite3 as sq
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
WEATHER_DATA_PATH = os.path.join(DATA_DIR, "part3", "chicago 2016-03-12 to 2016-04-09.csv")

# Task 1: Classification
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


def task1():
    df = pd.read_csv(os.path.join(DATA_DIR, "daily_activity.csv"))
    id_counts = df.groupby('Id').size().reset_index(name='Count')
    classified_df = classify_ids(id_counts)
    print(classified_df)


# Task 2: Database analysis
def task2(conn):
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

    return df_merged


# Task 3: Summaries
def get_daily_activity(conn, user_id=None, start_date=None, end_date=None):
    query = "SELECT * FROM daily_activity WHERE 1=1"
    params = []

    if user_id:
        query += " AND Id = ?"
        params.append(user_id)

    if start_date:
        query += " AND ActivityDate >= ?"
        params.append(start_date)

    if end_date:
        query += " AND ActivityDate <= ?"
        params.append(end_date)

    return pd.read_sql_query(query, conn, params=params)

def numerical_summary(df):
    return {
        "Total Steps": df["TotalSteps"].sum(),
        "Average Daily Steps": df["TotalSteps"].mean(),
        "Total Calories": df["Calories"].sum(),
        "Average Sedentary Minutes": df["SedentaryMinutes"].mean(),
        "Average Very Active Minutes": df["VeryActiveMinutes"].mean()
    }

def daily_summary(df):
    return (
        df.groupby("ActivityDate")
          .agg({
              "TotalSteps": "sum",
              "Calories": "sum",
              "VeryActiveMinutes": "sum",
              "SedentaryMinutes": "sum"
          })
          .reset_index()
    )

def get_dashboard_data(conn, user_id, start_date=None, end_date=None):
    df = get_daily_activity(conn, user_id, start_date, end_date)

    daily = daily_summary(df)
    numbers = numerical_summary(df)

    return {
        "daily_data": daily,
        "numerical_summary": numbers
    }

def task3(conn, user_id, start_date=None, end_date=None):
    dashboard_data = get_dashboard_data(conn, user_id, start_date, end_date)

    print("\nNumerical Summary:")
    for key, value in dashboard_data["numerical_summary"].items():
        print(f"{key}: {value:.2f}")

    print("\nDaily Summary Head:")
    print(dashboard_data["daily_data"].head())

    return dashboard_data
    

    

# Task 4: 4-hour block averages
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
    df_steps = pd.read_sql_query("SELECT Id, ActivityHour, StepTotal FROM hourly_steps", conn)
    df_steps['ActivityHour'] = pd.to_datetime(df_steps['ActivityHour'])
    df_steps['block'] = df_steps['ActivityHour'].dt.hour.apply(assign_block)
    df_steps['date'] = df_steps['ActivityHour'].dt.date
    steps_per_block = df_steps.groupby(['Id', 'date', 'block'])['StepTotal'].sum().reset_index()
    avg_steps = steps_per_block.groupby('block')['StepTotal'].mean().reindex(BLOCKS)

    df_cals = pd.read_sql_query("SELECT Id, ActivityHour, Calories FROM hourly_calories", conn)
    df_cals['ActivityHour'] = pd.to_datetime(df_cals['ActivityHour'])
    df_cals['block'] = df_cals['ActivityHour'].dt.hour.apply(assign_block)
    df_cals['date'] = df_cals['ActivityHour'].dt.date
    cals_per_block = df_cals.groupby(['Id', 'date', 'block'])['Calories'].sum().reset_index()
    avg_cals = cals_per_block.groupby('block')['Calories'].mean().reindex(BLOCKS)

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


def task4(conn):
    fig = plot_block_averages(conn)
    plt.show()


# Task 5: Heart rate and exercise intensity
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


def task5(conn):
    fig = plot_individual(conn, 4020332650)
    plt.show()


# Task 6: Correlation weather factors and activity levels
def plot_weather_activity_correlation(conn):
    weather_df = pd.read_csv(WEATHER_DATA_PATH)
    
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
    weather_df['date'] = weather_df['datetime'].dt.date

    weather_cols = ['temp', 'feelslike', 'humidity', 'precip', 'windspeed', 'cloudcover', 'visibility', 'uvindex']
    weather_agg = weather_df.groupby('date')[weather_cols].mean().reset_index()

    activity_cols = ['TotalSteps', 'TotalDistance', 'Calories', 'VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes']
    query = f"""
        SELECT ActivityDate, {', '.join(activity_cols)}
        FROM daily_activity
    """
    activity_df = pd.read_sql_query(query, conn)
    activity_df['date'] = pd.to_datetime(activity_df['ActivityDate']).dt.date
    activity_agg = activity_df.groupby('date')[activity_cols].mean().reset_index()

    merged_df = pd.merge(activity_agg, weather_agg, on='date', how='inner')

    correlation_df = merged_df[activity_cols + weather_cols].corr()
    plt.figure(figsize=(12, 8))
    im = plt.imshow(correlation_df.loc[weather_cols, activity_cols], cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, label='Correlation Coefficient')
    plt.xticks(range(len(activity_cols)), activity_cols, rotation=45, ha='right')
    plt.yticks(range(len(weather_cols)), weather_cols)
    plt.title('Correlation between Weather Factors and Activity Levels')
    plt.tight_layout()
    plt.show()

    return correlation_df

def plot_windspeed_activity(conn, bins=10):
    weather_df = pd.read_csv(WEATHER_DATA_PATH)
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
    weather_df['date'] = weather_df['datetime'].dt.date

    wind_agg = weather_df.groupby('date')['windspeed'].mean().reset_index()

    activity_cols = ['TotalSteps', 'TotalDistance', 'Calories', 'VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes']
    query = f"SELECT ActivityDate, {', '.join(activity_cols)} FROM daily_activity"
    activity_df = pd.read_sql_query(query, conn)
    activity_df['date'] = pd.to_datetime(activity_df['ActivityDate']).dt.date
    activity_agg = activity_df.groupby('date')[activity_cols].mean().reset_index()

    merged = pd.merge(activity_agg, wind_agg, on='date', how='inner')

    merged['windspeed_bin'] = pd.cut(merged['windspeed'], bins=bins)
    agg_map = {c: 'mean' for c in activity_cols}
    agg_map['windspeed'] = 'mean'
    binned = merged.groupby('windspeed_bin').agg(agg_map).dropna()

    bin_centers = []
    for interval in binned.index.categories:
        left = interval.left
        right = interval.right
        bin_centers.append((left + right) / 2.0)
    bin_centers = np.array(bin_centers)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    for i, col in enumerate(activity_cols):
        ax.plot(bin_centers, binned[col].values, label=col, color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel('Windspeed (binned, mean per day)')
    ax.set_ylabel('Activity metric (mean per day)')
    ax.set_title('Activity levels vs Windspeed (binned)')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return fig, binned


def task6(conn):
    plot_weather_activity_correlation(conn)
    plot_windspeed_activity(conn, bins=5)

# Task 7: Weight log missing value imputation
def impute_weight_log(conn):
    df = pd.read_sql_query("SELECT CAST(Id AS INTEGER) AS Id, Date, WeightKg, WeightPounds, Fat, BMI, IsManualReport FROM weight_log", conn)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')

    # Fix WeightKg / WeightPounds using conversion
    KG_TO_LBS = 2.20462
    missing_kg = df['WeightKg'].isna() & df['WeightPounds'].notna()
    df.loc[missing_kg, 'WeightKg'] = df.loc[missing_kg, 'WeightPounds'] / KG_TO_LBS

    missing_lbs = df['WeightPounds'].isna() & df['WeightKg'].notna()
    df.loc[missing_lbs, 'WeightPounds'] = df.loc[missing_lbs, 'WeightKg'] * KG_TO_LBS

    # Infer height per user and fill missing BMI
    df['height_m'] = np.sqrt(df['WeightKg'] / df['BMI'])
    user_height = df.groupby('Id')['height_m'].mean()

    def fill_bmi(row):
        if pd.isna(row['BMI']):
            h = user_height.get(row['Id'])
            if h and not np.isnan(h):
                return row['WeightKg'] / (h ** 2)
        return row['BMI']

    df['BMI'] = df.apply(fill_bmi, axis=1)
    df = df.drop(columns=['height_m'])

    # Fill missing Fat using Deurenberg-inspired BMI formula
    # Full Deurenberg formula requires age & sex which are unavailable,
    # so we use Fat% ≈ 1.20 * BMI + offset, where the offset is calibrated from the known Fat observations to anchor estimates to actual data.
    known_fat = df[df['Fat'].notna()]
    missing_fat = df['Fat'].isna()

    if len(known_fat) >= 1:
        offset = (known_fat['Fat'] - 1.20 * known_fat['BMI']).mean()
        df.loc[missing_fat, 'Fat'] = (1.20 * df.loc[missing_fat, 'BMI'] + offset).clip(lower=0, upper=60)
    else:
        df.loc[missing_fat, 'Fat'] = df['Fat'].mean()

    print("=== Missing values after imputation ===")
    print(df.isnull().sum())
    print()
    print(df)

    return df

# main
def main():
    conn = sq.connect(os.path.join(DATA_DIR, "fitbit_database.db"))

    task1()
    df_merged = task2(conn)
    task3(conn, 4020332650)
    task4(conn)
    task5(conn)
    task6(conn)
    impute_weight_log(conn)
    
    conn.close()

if __name__ == "__main__":
    main()
