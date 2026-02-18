import pandas as pd
import sqlite3 as sq
from sklearn.linear_model import LinearRegression

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

print(df_merged)

    