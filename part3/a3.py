import pandas as pd


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

print(df)