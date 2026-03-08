import os
import sys
import sqlite3
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

# import funcs from a4
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "part4"))
import a4

# Database function
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "fitbit_database.db")

@st.cache_resource
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

conn = get_conn()   # global connection object for all data loading functions

# Data Loading
# Wrap loading functions with caching so that it doesnt take forever

## Helper Functions
@st.cache_data
def loadAllActivity():    # all daily activity data and parsed date
    df = pd.read_sql_query("SELECT * FROM daily_activity", conn)
    df["ActivityDate"] = pd.to_datetime(df["ActivityDate"], format="mixed").dt.normalize()
    df["Id"] = df["Id"].astype(int)
    return df

## Functions for specific view
@st.cache_data
def loadUserIds():    # load all user ids for dropdown selection
    df = pd.read_sql_query(
        "SELECT DISTINCT CAST(Id AS INTEGER) AS Id FROM daily_activity ORDER BY Id", conn
    )
    return df["Id"].tolist()

@st.cache_data
def loadHrUserIds(): # load heart rate user ids
    df = pd.read_sql_query(
        "SELECT DISTINCT CAST(Id AS INTEGER) AS Id FROM heart_rate ORDER BY Id", conn
    )
    return set(df["Id"].tolist())

@st.cache_data
def loadDashboardData(user_id, start, end):   # load data and make summaries for individual
    df = loadAllActivity()
    df = df[df["Id"] == int(user_id)]
    df = df[(df["ActivityDate"].dt.date >= start) & (df["ActivityDate"].dt.date <= end)].copy()
    df["ActivityDate"] = df["ActivityDate"].dt.strftime("%Y-%m-%d")
    return {
        "daily_data":        a4.daily_summary(df),
        "numerical_summary": a4.numerical_summary(df),
    }

@st.cache_data
def loadAllSleep():   # load all sleep data
    df = pd.read_sql_query("""
        SELECT CAST(Id AS INTEGER) AS Id, date, logId, COUNT(*) AS sleep_minutes
        FROM minute_sleep
        GROUP BY Id, logId
    """, conn)
    df["date"] = pd.to_datetime(df["date"], format="mixed").dt.normalize()
    return df

@st.cache_data
def loadSleepAnalysis(user_id, start, end):   # load and merge sleep and activity data for correlation and comparison analysis
    act = loadAllActivity()
    act = act[act["Id"] == int(user_id)]
    act = act[(act["ActivityDate"].dt.date >= start) & (act["ActivityDate"].dt.date <= end)].copy()
    act["active_minutes"] = act["VeryActiveMinutes"] + act["FairlyActiveMinutes"] + act["LightlyActiveMinutes"]
    act["is_weekend"] = act["ActivityDate"].dt.weekday >= 5

    slp = loadAllSleep()
    slp = slp[slp["Id"] == int(user_id)].copy()

    merged = pd.merge(
        slp[["date", "sleep_minutes"]],
        act[["ActivityDate", "TotalSteps", "Calories", "SedentaryMinutes",
             "active_minutes", "VeryActiveMinutes", "is_weekend"]],
        left_on="date", right_on="ActivityDate", how="inner"
    )
    return merged

@st.cache_data
def loadHeartRate(user_id, start, end):   # load heart rate data
    df = pd.read_sql_query("""
        SELECT Time, Value AS heart_rate
        FROM heart_rate
        WHERE CAST(Id AS INTEGER) = ?
        ORDER BY Time
    """, conn, params=(int(user_id),))
    if df.empty:
        return df
    df["Time"] = pd.to_datetime(df["Time"], format="mixed")
    df = df[(df["Time"].dt.date >= start) & (df["Time"].dt.date <= end)]
    return df

@st.cache_data
def loadBlockAverages():  # just do whatever a4 does
    return a4.compute_block_averages(conn)

## Functions for general view
@st.cache_data
def loadGeneralStats():   # statistics of all users for the overview page
    act = loadAllActivity()

    total_users = act["Id"].nunique()
    total_days = act["ActivityDate"].nunique()
    avg_steps = act["TotalSteps"].mean()
    avg_calories = act["Calories"].mean()
    avg_sedentary = act["SedentaryMinutes"].mean()
    avg_active = (act["VeryActiveMinutes"] + act["FairlyActiveMinutes"] + act["LightlyActiveMinutes"]).mean()

    # daily average steps for line chart
    daily_avg = (
        act.groupby("ActivityDate")["TotalSteps"].mean()
        .reset_index()
        .rename(columns={"TotalSteps": "Avg Steps (all users)"})
        .set_index("ActivityDate")
    )

    # average activity type per user class
    id_counts  = act.groupby("Id").size().reset_index(name="Count")
    id_classes = a4.classify_ids(id_counts)
    act_c = act.merge(id_classes, on="Id", how="left")
    class_avg = act_c.groupby("Class", observed=True)[["TotalSteps", "Calories", "SedentaryMinutes"]].mean()

    return {
        "total_users":   total_users,
        "total_days":    total_days,
        "avg_steps":     avg_steps,
        "avg_calories":  avg_calories,
        "avg_sedentary": avg_sedentary,
        "avg_active":    avg_active,
        "daily_avg":     daily_avg,
        "class_avg":     class_avg,
    }

# Page Layout

st.set_page_config(page_title="Fitbit Analytics Dashboard", layout="wide")
st.title("Fitbit Analytics Dashboard")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Individual"])

DATA_START = date(2016, 3, 12)
DATA_END   = date(2016, 4, 9)

## Page 1: General Overview
if page == "Overview":
    st.header("Overview of All Participants")
    st.caption(f"Date range: {DATA_START} to {DATA_END}")

    gs = loadGeneralStats()

    # numerical summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Participants", f"{gs['total_users']}")
    c2.metric("Average Daily Steps", f"{gs['avg_steps']:.0f}")
    c3.metric("Average Calories per Day", f"{gs['avg_calories']:.0f}")
    c4.metric("Average Active Minutes per Day", f"{gs['avg_active']:.0f}")

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Average Daily Steps Over Time")
        st.line_chart(gs["daily_avg"])

    with col_r:
        st.subheader("Key Metrics per User Class")
        st.dataframe(gs["class_avg"].round(1), width="stretch")

    st.divider()
    st.subheader("Activity Type Breakdown (Pie Chart)")
    act_all = loadAllActivity()
    breakdown = act_all[["VeryActiveMinutes", "FairlyActiveMinutes",
                          "LightlyActiveMinutes", "SedentaryMinutes"]].mean()
    fig, ax = plt.subplots(figsize=(5, 5))
    total = breakdown.values.sum()
    legend_labels = [
        f"{name} ({val / total * 100:.1f}%)"
        for name, val in zip(breakdown.index, breakdown.values)
    ]
    wedges, _ = ax.pie(
        breakdown.values,
        labels=None,
        autopct=None,
        startangle=90,
        colors=["#e74c3c", "#e67e22", "#3498db", "#95a5a6"],
    )
    ax.legend(
        wedges,
        legend_labels,
        title="Activity Type",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    ax.axis("equal")
    plt.tight_layout()
    st.pyplot(fig)

## Page 2: Individual User View
else:
    st.sidebar.header("Filters")

    user_id = st.sidebar.selectbox("Select User ID", loadUserIds())

    # validate selectable date range (no breaking the rules allowed)
    start_date = st.sidebar.date_input(
        "Start Date", value=DATA_START, min_value=DATA_START, max_value=DATA_END
    )
    end_date = st.sidebar.date_input(
        "End Date", value=DATA_END, min_value=DATA_START, max_value=DATA_END
    )
    time_block = st.sidebar.selectbox(
        "Time of Day",
        ["All Day", "0-4", "4-8", "8-12", "12-16", "16-20", "20-24"]
    )

    # if exists load hr data
    hr_users = loadHrUserIds()
    if user_id in hr_users:
        st.sidebar.success("HR data available")
    else:
        st.sidebar.warning("NO HR data available (try e.g. 2022484408 or 6117666160)")

    st.header(f"Individual View for User {user_id}")

    # key numerical stats
    dashboard = loadDashboardData(user_id, start_date, end_date)
    nums = dashboard["numerical_summary"]
    daily_df = dashboard["daily_data"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Steps", f"{int(nums['Total Steps']):,}")
    col2.metric("Average Daily Steps", f"{nums['Average Daily Steps']:,.0f}")
    col3.metric("Total Calories", f"{int(nums['Total Calories']):,}")
    col4.metric("Average Sedentary Minutes", f"{nums['Average Sedentary Minutes']:,.0f}")

    # different tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Daily Activity", "Sleep Analysis", "Heart Rate", "4-Hour Blocks"
    ])

    # tab 1 => daily steps and calories trend
    with tab1:
        st.subheader("Daily Steps & Calories Trend")
        if daily_df.empty:
            st.info("No activity data for this selection.")
        else:
            daily_df["ActivityDate"] = pd.to_datetime(daily_df["ActivityDate"])
            st.line_chart(daily_df.set_index("ActivityDate")[["TotalSteps", "Calories"]])

    # tab 2 => sleep duration correlation and weekend vs weekday comparison
    with tab2:
        st.subheader("Sleep Duration Analysis")
        sleep_df = loadSleepAnalysis(user_id, start_date, end_date)

        if sleep_df.empty:
            st.info("No sleep data available for this user / date range.")
        else:   # correlation of sleep duration with activity variables
            vars_of_interest = ["TotalSteps", "active_minutes", "VeryActiveMinutes",
                                 "SedentaryMinutes", "Calories"]
            corr = sleep_df[["sleep_minutes"] + vars_of_interest].corr()["sleep_minutes"].drop("sleep_minutes")
            corr_df = corr.reset_index()
            corr_df.columns = ["Variable", "Correlation with Sleep (min)"]
            corr_df = corr_df.sort_values("Correlation with Sleep (min)", ascending=False)

            st.markdown("**Correlation of activity variables with sleep duration**")
            corr_df["Correlation with Sleep (min)"] = corr_df["Correlation with Sleep (min)"].round(3)
            st.dataframe(corr_df, width="stretch")

            st.divider()

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Sleep vs Active Minutes**")
                st.scatter_chart(sleep_df, x="active_minutes", y="sleep_minutes")

            with col_b:
                st.markdown("**Sleep vs Sedentary Minutes**")
                st.scatter_chart(sleep_df, x="SedentaryMinutes", y="sleep_minutes")

            st.divider()

            # weekend vs weekday
            wk = sleep_df.groupby("is_weekend")["sleep_minutes"].agg(["mean", "median", "count"])
            wk.index = ["Weekday", "Weekend"]
            wk.columns = ["Mean Sleep (min)", "Median Sleep (min)", "Observations"]
            st.markdown("**Weekend vs Weekday Sleep Duration**")
            st.dataframe(wk.round(1), width="stretch")

    # tab 3 => heart rate data
    with tab3:
        st.subheader("Heart Rate Over Time")
        hr_df = loadHeartRate(user_id, start_date, end_date)
        if hr_df.empty:
            st.info(
                f"No heart rate data for user {user_id}. "
                "Only 14 of the 35 study participants wore a heart rate sensor."
                "Try one of these users: 2022484408, 2026352035, 4020332650, 5553957443, 6117666160."
            )
        else:
            hr_df["Time"] = pd.to_datetime(hr_df["Time"])
            st.line_chart(hr_df.set_index("Time")["heart_rate"])

    # tab 4 => 4 hour blocks
    with tab4:
        st.subheader("Average Steps & Calories per 4-Hour Block")
        avg_steps, avg_cals, avg_sleep = loadBlockAverages()
        block_df = pd.DataFrame({
            "Avg Steps":    avg_steps.values,
            "Avg Calories": avg_cals.values,
        }, index=avg_steps.index)
        st.bar_chart(block_df)