import os
import sqlite3
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "part4"))
import a4

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "fitbit_database.db")
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ACTIVITY_COLORS = {
    "SedentaryMinutes": "#c31e15",
    "LightlyActiveMinutes": "#ffc516",
    "FairlyActiveMinutes": "#0871b2",
    "VeryActiveMinutes": "#00a84c",
}
CLASS_COLORS = {
    "Light user": "#9ecae1",
    "Moderate user": "#f6ae2d",
    "Heavy user": "#d1495b",
}


@st.cache_resource
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


CONN = get_conn()


@st.cache_data
def load_daily_activity():
    df = pd.read_sql_query("SELECT * FROM daily_activity", CONN)
    df["Id"] = df["Id"].astype(int)
    df["ActivityDate"] = pd.to_datetime(df["ActivityDate"], format="mixed").dt.normalize()
    df["active_minutes"] = (
        df["VeryActiveMinutes"] + df["FairlyActiveMinutes"] + df["LightlyActiveMinutes"]
    )
    df["weekday"] = df["ActivityDate"].dt.day_name()
    return df


@st.cache_data
def load_sleep_daily():
    df = pd.read_sql_query(
        """
        SELECT CAST(Id AS INTEGER) AS Id, date, logId
        FROM minute_sleep
        """,
        CONN,
    )
    df["datetime"] = pd.to_datetime(df["date"], format="mixed")
    sleep_daily = (
        df.groupby(["Id", "logId"], as_index=False)
        .agg(
            sleep_minutes=("date", "size"),
            SleepDate=("datetime", lambda s: s.min().normalize()),
        )
    )
    sleep_daily["weekday"] = sleep_daily["SleepDate"].dt.day_name()
    return sleep_daily


@st.cache_data
def load_hourly_steps():
    df = pd.read_sql_query(
        "SELECT CAST(Id AS INTEGER) AS Id, ActivityHour, StepTotal FROM hourly_steps",
        CONN,
    )
    df["ActivityHour"] = pd.to_datetime(df["ActivityHour"], format="mixed")
    df["date"] = df["ActivityHour"].dt.normalize()
    df["hour"] = df["ActivityHour"].dt.hour
    df["weekday"] = df["ActivityHour"].dt.day_name()
    df["block"] = df["hour"].apply(a4.assign_block)
    return df


@st.cache_data
def load_hourly_intensity():
    df = pd.read_sql_query(
        "SELECT CAST(Id AS INTEGER) AS Id, ActivityHour, TotalIntensity FROM hourly_intensity",
        CONN,
    )
    df["ActivityHour"] = pd.to_datetime(df["ActivityHour"], format="mixed")
    df["date"] = df["ActivityHour"].dt.normalize()
    df["hour"] = df["ActivityHour"].dt.hour
    return df


@st.cache_data
def load_daily_intensity():
    df = load_hourly_intensity()
    return (
        df.groupby(["Id", "date"], as_index=False)["TotalIntensity"]
        .sum()
        .rename(columns={"date": "ActivityDate", "TotalIntensity": "daily_intensity"})
    )


@st.cache_data
def load_user_classes():
    return a4.classify_users_by_steps(load_daily_activity())


@st.cache_data
def load_user_summary():
    activity = load_daily_activity()
    sleep = load_sleep_daily()
    classes = load_user_classes()[["Id", "Class", "avg_daily_steps"]]

    activity_summary = (
        activity.groupby("Id", as_index=False)
        .agg(
            avg_daily_steps=("TotalSteps", "mean"),
            avg_daily_calories=("Calories", "mean"),
            avg_active_minutes=("active_minutes", "mean"),
        )
    )
    sleep_summary = (
        sleep.groupby("Id", as_index=False)["sleep_minutes"]
        .mean()
        .rename(columns={"sleep_minutes": "avg_daily_sleep"})
    )

    summary = activity_summary.merge(sleep_summary, on="Id", how="left")
    summary = summary.merge(classes[["Id", "Class"]], on="Id", how="left")
    return summary.sort_values("avg_daily_steps", ascending=False).reset_index(drop=True)


def get_date_bounds():
    activity = load_daily_activity()
    return activity["ActivityDate"].min().date(), activity["ActivityDate"].max().date()


def filter_daily_activity(user_choice, start_date, end_date):
    df = load_daily_activity()
    mask = (df["ActivityDate"].dt.date >= start_date) & (df["ActivityDate"].dt.date <= end_date)
    if user_choice != "All Users":
        mask &= df["Id"] == int(user_choice)
    return df.loc[mask].copy()


def filter_sleep_daily(user_choice, start_date, end_date):
    df = load_sleep_daily()
    mask = (df["SleepDate"].dt.date >= start_date) & (df["SleepDate"].dt.date <= end_date)
    if user_choice != "All Users":
        mask &= df["Id"] == int(user_choice)
    return df.loc[mask].copy()


def filter_hourly_steps(user_choice, start_date, end_date):
    df = load_hourly_steps()
    mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
    if user_choice != "All Users":
        mask &= df["Id"] == int(user_choice)
    return df.loc[mask].copy()


def build_activity_sleep_dataset(start_date, end_date):
    activity = filter_daily_activity("All Users", start_date, end_date)
    sleep = filter_sleep_daily("All Users", start_date, end_date)
    intensity = load_daily_intensity()
    intensity = intensity[
        (intensity["ActivityDate"].dt.date >= start_date)
        & (intensity["ActivityDate"].dt.date <= end_date)
    ].copy()

    merged = activity.merge(
        sleep[["Id", "SleepDate", "sleep_minutes"]],
        left_on=["Id", "ActivityDate"],
        right_on=["Id", "SleepDate"],
        how="inner",
    )
    merged = merged.merge(intensity, on=["Id", "ActivityDate"], how="left")
    merged = merged.merge(load_user_classes()[["Id", "Class"]], on="Id", how="left")
    return merged.sort_values(["Id", "ActivityDate"]).reset_index(drop=True)


def format_user_option(user_id):
    return "All Users" if user_id == "All Users" else f"User {user_id}"


def ensure_specific_user(user_choice, page_name):
    if user_choice == "All Users":
        st.info(f"{page_name} requires a specific user. Select a user ID in the sidebar.")
        return False
    return True


def compute_filtered_user_summary(start_date, end_date):
    activity = filter_daily_activity("All Users", start_date, end_date)
    sleep = filter_sleep_daily("All Users", start_date, end_date)
    classes = load_user_classes()[["Id", "Class"]]

    activity_summary = (
        activity.groupby("Id", as_index=False)
        .agg(
            avg_daily_steps=("TotalSteps", "mean"),
            avg_daily_calories=("Calories", "mean"),
        )
    )
    sleep_summary = (
        sleep.groupby("Id", as_index=False)["sleep_minutes"]
        .mean()
        .rename(columns={"sleep_minutes": "avg_daily_sleep"})
    )
    return (
        activity_summary.merge(sleep_summary, on="Id", how="left")
        .merge(classes, on="Id", how="left")
        .sort_values("avg_daily_steps", ascending=False)
        .reset_index(drop=True)
    )


def build_dual_axis_trend(daily_activity):
    trend = (
        daily_activity.groupby("ActivityDate", as_index=False)
        .agg(avg_steps=("TotalSteps", "mean"), avg_calories=("Calories", "mean"))
        .sort_values("ActivityDate")
    )
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax2 = ax1.twinx()

    ax1.plot(trend["ActivityDate"], trend["avg_steps"], color="#1f77b4", linewidth=2, label="Avg Steps")
    ax2.plot(trend["ActivityDate"], trend["avg_calories"], color="#ff7f0e", linewidth=2, label="Avg Calories")

    ax1.set_title("Average Daily Steps and Calories Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Steps", color="#1f77b4")
    ax2.set_ylabel("Calories", color="#ff7f0e")
    ax1.grid(True, linestyle="--", alpha=0.4)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def build_activity_distribution(activity_df, title):
    breakdown = (
        activity_df.groupby("weekday")[
            ["SedentaryMinutes", "LightlyActiveMinutes", "FairlyActiveMinutes", "VeryActiveMinutes"]
        ]
        .mean()
        .reindex(DAY_ORDER)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(breakdown))
    for col in breakdown.columns:
        ax.bar(
            breakdown.index,
            breakdown[col].values,
            bottom=bottom,
            label=col.replace("Minutes", ""),
            color=ACTIVITY_COLORS[col],
        )
        bottom += breakdown[col].values

    ax.set_title(title)
    ax.set_ylabel("Average Minutes")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    return fig


def build_active_hours_heatmap(hourly_steps_df):
    heatmap = (
        hourly_steps_df.groupby(["weekday", "hour"], as_index=False)["StepTotal"]
        .mean()
        .pivot(index="weekday", columns="hour", values="StepTotal")
        .reindex(DAY_ORDER)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(11, 4.8))
    image = ax.imshow(heatmap.values, aspect="auto", cmap="YlOrRd")
    ax.set_title("Most Active Hours Across the Week")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_yticks(range(len(DAY_ORDER)))
    ax.set_yticklabels(DAY_ORDER)
    fig.colorbar(image, ax=ax, label="Average Steps")
    plt.tight_layout()
    return fig


def build_hourly_bar(hourly_steps_df, title):
    hourly = hourly_steps_df.groupby("hour", as_index=False)["StepTotal"].mean()
    hourly = hourly.set_index("hour").reindex(range(24), fill_value=0).reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(hourly["hour"], hourly["StepTotal"], color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Steps")
    ax.set_xticks(range(24))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def build_grouped_day_comparison(user_series, overall_series, title, y_label):
    compare = pd.DataFrame({"Selected User": user_series, "Population Average": overall_series}).reindex(DAY_ORDER)
    x = np.arange(len(compare.index))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - width / 2, compare["Selected User"].fillna(0), width=width, label="Selected User", color="#1f77b4")
    ax.bar(x + width / 2, compare["Population Average"].fillna(0), width=width, label="Population Average", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(compare.index, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def build_regression_stats(x, y):
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 2:
        return None

    x_vals = clean["x"].to_numpy(dtype=float)
    y_vals = clean["y"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    predictions = slope * x_vals + intercept
    ss_res = np.sum((y_vals - predictions) ** 2)
    ss_tot = np.sum((y_vals - y_vals.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return {
        "x": x_vals,
        "y": y_vals,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "predictions": predictions,
    }


def build_regression_plot(dataset, x_col, x_label, selected_user=None):
    stats = build_regression_stats(dataset[x_col], dataset["sleep_minutes"])
    if stats is None:
        return None, None

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(stats["x"], stats["y"], alpha=0.35, color="#7f8c8d", label="All users")

    x_line = np.linspace(stats["x"].min(), stats["x"].max(), 200)
    y_line = stats["slope"] * x_line + stats["intercept"]

    n = len(stats["x"])
    mean_x = stats["x"].mean()
    ssx = np.sum((stats["x"] - mean_x) ** 2)
    residual_se = np.sqrt(np.sum((stats["y"] - stats["predictions"]) ** 2) / max(n - 2, 1))
    if ssx > 0 and n > 2:
        se_line = residual_se * np.sqrt(1 / n + ((x_line - mean_x) ** 2) / ssx)
        margin = 1.96 * se_line
        ax.fill_between(x_line, y_line - margin, y_line + margin, color="#5aa9e6", alpha=0.2, label="95% band")

    ax.plot(x_line, y_line, color="#1f77b4", linewidth=2, label="Regression line")

    if selected_user is not None:
        user_df = dataset[dataset["Id"] == int(selected_user)]
        if not user_df.empty:
            ax.scatter(
                user_df[x_col],
                user_df["sleep_minutes"],
                color="#d1495b",
                edgecolor="white",
                linewidth=0.5,
                s=55,
                label=f"User {selected_user}",
            )

    ax.set_title(f"{x_label} vs Sleep Duration")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Sleep Duration (minutes)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    return fig, stats


def regression_interpretation(slope, r2, x_label):
    direction = "higher" if slope > 0 else "lower"
    strength = "weak"
    if r2 >= 0.5:
        strength = "strong"
    elif r2 >= 0.2:
        strength = "moderate"

    return (
        f"The relationship is {strength}: {direction} {x_label.lower()} tends to align with "
        f"{'more' if slope > 0 else 'less'} sleep, although the spread in the data is still substantial."
    )


def build_sleep_pattern_chart(sleep_df, selected_user):
    sleep_df = sleep_df.copy()
    sleep_df["weekday"] = sleep_df["SleepDate"].dt.day_name()

    fig, ax = plt.subplots(figsize=(10, 4.8))
    if selected_user == "All Users":
        sleep_df = sleep_df.merge(load_user_classes()[["Id", "Class"]], on="Id", how="left")
        grouped = (
            sleep_df.groupby(["weekday", "Class"])["sleep_minutes"]
            .agg(["mean", "std"])
            .reset_index()
        )
        x = np.arange(len(DAY_ORDER))
        width = 0.25
        for idx, class_name in enumerate(a4.CLASS_ORDER):
            class_slice = grouped[grouped["Class"] == class_name].set_index("weekday").reindex(DAY_ORDER)
            ax.bar(
                x + (idx - 1) * width,
                class_slice["mean"].fillna(0),
                yerr=class_slice["std"].fillna(0),
                width=width,
                label=class_name,
                color=CLASS_COLORS[class_name],
                capsize=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(DAY_ORDER, rotation=30, ha="right")
        ax.legend()
    else:
        grouped = sleep_df.groupby("weekday")["sleep_minutes"].agg(["mean", "std"]).reindex(DAY_ORDER)
        ax.bar(
            grouped.index,
            grouped["mean"].fillna(0),
            yerr=grouped["std"].fillna(0),
            color="#1f77b4",
            capsize=4,
        )
        ax.tick_params(axis="x", rotation=30)

    ax.set_title("Day-of-Week Sleep Patterns")
    ax.set_ylabel("Average Sleep Duration (minutes)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def build_class_distribution_chart(summary_df):
    counts = summary_df["Class"].value_counts().reindex(a4.CLASS_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    colors = [CLASS_COLORS[class_name] for class_name in a4.CLASS_ORDER]
    wedges, _ = ax.pie(
        counts.values,
        labels=None,
        startangle=90,
        colors=colors,
        wedgeprops={"width": 0.42},
    )
    ax.legend(
        wedges,
        [f"{label} ({int(value)})" for label, value in zip(a4.CLASS_ORDER, counts.values)],
        title="User Class",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )
    ax.set_title("User Class Distribution")
    plt.tight_layout()
    return fig


def render_page_one():
    st.header("Page 1: General Overview")

    daily_activity = load_daily_activity()
    sleep_daily = load_sleep_daily()
    user_summary = load_user_summary()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users", f"{daily_activity['Id'].nunique()}")
    col2.metric("Average Daily Steps", f"{daily_activity['TotalSteps'].mean():,.0f}")
    col3.metric("Average Daily Sleep", f"{sleep_daily['sleep_minutes'].mean():,.0f} min")
    col4.metric("Average Daily Calories", f"{daily_activity['Calories'].mean():,.0f}")

    top_left, top_right = st.columns(2)
    with top_left:
        st.pyplot(build_dual_axis_trend(daily_activity), clear_figure=True)
    with top_right:
        st.pyplot(build_class_distribution_chart(user_summary), clear_figure=True)

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.pyplot(
            build_activity_distribution(daily_activity, "Average Activity Level Distribution by Weekday"),
            clear_figure=True,
        )
    with bottom_right:
        st.pyplot(build_active_hours_heatmap(load_hourly_steps()), clear_figure=True)

    display_table = user_summary.rename(
        columns={
            "avg_daily_steps": "Avg Daily Steps",
            "avg_daily_calories": "Avg Daily Calories",
            "avg_daily_sleep": "Avg Daily Sleep",
        }
    )[["Id", "Avg Daily Steps", "Avg Daily Calories", "Avg Daily Sleep", "Class"]]
    style = display_table.style.background_gradient(
        cmap="Blues",
        subset=["Avg Daily Steps", "Avg Daily Calories", "Avg Daily Sleep"],
    ).format(
        {
            "Average Daily Steps": "{:,.0f}",
            "Average Daily Calories": "{:,.0f}",
            "Average Daily Sleep": "{:,.0f}",
        }
    )
    st.subheader("Per-User Summary")
    st.dataframe(style, use_container_width=True, hide_index=True)


def render_page_two(user_choice, start_date, end_date):
    st.header("Page 2: Individual Summary")
    if not ensure_specific_user(user_choice, "Page 2"):
        return

    activity = filter_daily_activity(user_choice, start_date, end_date)
    sleep = filter_sleep_daily(user_choice, start_date, end_date)
    overall_activity = filter_daily_activity("All Users", start_date, end_date)
    overall_sleep = filter_sleep_daily("All Users", start_date, end_date)
    hourly_steps = filter_hourly_steps(user_choice, start_date, end_date)

    classes = load_user_classes().set_index("Id")
    user_class = classes.loc[int(user_choice), "Class"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("User Class", user_class.replace(" user", ""))
    col2.metric(
        "Average Daily Steps",
        f"{activity['TotalSteps'].mean():,.0f}",
        delta=f"{activity['TotalSteps'].mean() - overall_activity['TotalSteps'].mean():+.0f} vs average",
    )
    col3.metric(
        "Average Daily Sleep",
        f"{sleep['sleep_minutes'].mean():,.0f} min" if not sleep.empty else "N/A",
        delta=(
            f"{sleep['sleep_minutes'].mean() - overall_sleep['sleep_minutes'].mean():+.0f} min vs average"
            if not sleep.empty and not overall_sleep.empty
            else None
        ),
    )
    col4.metric(
        "Average Daily Calories",
        f"{activity['Calories'].mean():,.0f}",
        delta=f"{activity['Calories'].mean() - overall_activity['Calories'].mean():+.0f} vs average",
    )

    st.subheader("Percentile Rank")
    summary = compute_filtered_user_summary(start_date, end_date)
    user_row = summary[summary["Id"] == int(user_choice)]
    if not user_row.empty:
        percentile_cols = st.columns(3)
        for idx, metric in enumerate(
            [
                ("avg_daily_steps", "Steps"),
                ("avg_daily_sleep", "Sleep"),
                ("avg_daily_calories", "Calories"),
            ]
        ):
            rank = summary[metric[0]].rank(pct=True)
            percentile = rank.loc[summary["Id"] == int(user_choice)].iloc[0] * 100
            percentile_cols[idx].metric(f"{metric[1]} Percentile", f"{percentile:.0f}th")

    merged = activity[["ActivityDate", "TotalSteps", "Calories"]].merge(
        sleep[["SleepDate", "sleep_minutes"]],
        left_on="ActivityDate",
        right_on="SleepDate",
        how="left",
    )

    st.subheader("Daily Trends (Steps, Calories and Sleep)")
    chart_cols = st.columns(3)
    with chart_cols[0]:
        st.line_chart(merged.set_index("ActivityDate")["TotalSteps"])
    with chart_cols[1]:
        st.line_chart(merged.set_index("ActivityDate")["Calories"])
    with chart_cols[2]:
        sleep_series = merged.set_index("ActivityDate")["sleep_minutes"]
        st.line_chart(sleep_series.dropna() if not sleep_series.dropna().empty else sleep_series)

    st.subheader("Day-of-Week Breakdown")
    user_steps = activity.groupby("weekday")["TotalSteps"].mean()
    overall_steps = overall_activity.groupby("weekday")["TotalSteps"].mean()
    user_sleep = sleep.groupby("weekday")["sleep_minutes"].mean()
    overall_sleep = overall_sleep.groupby("weekday")["sleep_minutes"].mean()
    breakdown_cols = st.columns(2)
    with breakdown_cols[0]:
        st.pyplot(
            build_grouped_day_comparison(
                user_steps, overall_steps, "Average Steps by Day: User vs Population", "Average Steps"
            ),
            clear_figure=True,
        )
    with breakdown_cols[1]:
        st.pyplot(
            build_grouped_day_comparison(
                user_sleep, overall_sleep, "Average Sleep by Day: User vs Population", "Sleep Minutes"
            ),
            clear_figure=True,
        )

    st.subheader("Most Active Hours")
    st.pyplot(build_hourly_bar(hourly_steps, f"Average Hourly Steps for User {user_choice}"), clear_figure=True)

    st.subheader("Activity Level Distribution")
    st.pyplot(
        build_activity_distribution(activity, f"Activity Mix Through the Week for User {user_choice}"),
        clear_figure=True,
    )


def render_page_three(user_choice, start_date, end_date):
    st.header("Page 3: Sleep Analysis")

    dataset = build_activity_sleep_dataset(start_date, end_date)
    sleep_view = filter_sleep_daily(user_choice, start_date, end_date)
    if user_choice != "All Users":
        dataset = dataset[dataset["Id"] == int(user_choice)].copy()

    if dataset.empty:
        st.info("No joined activity and sleep records are available for this selection.")
        return

    st.subheader("Regression Panel")
    variable_map = {
        "Total Steps": "TotalSteps",
        "Active Minutes": "active_minutes",
        "Sedentary Minutes": "SedentaryMinutes",
        "Calories": "Calories",
        "Intensity": "daily_intensity",
    }
    selected_label = st.selectbox("Choose the explanatory variable", list(variable_map.keys()))
    regression_base = build_activity_sleep_dataset(start_date, end_date)
    fig, stats = build_regression_plot(
        regression_base,
        variable_map[selected_label],
        selected_label,
        None if user_choice == "All Users" else user_choice,
    )

    if fig is None:
        st.info("Not enough data to fit the regression line.")
    else:
        st.pyplot(fig, clear_figure=True)
        metric_cols = st.columns(3)
        metric_cols[0].metric("Slope", f"{stats['slope']:.3f}")
        metric_cols[1].metric("Intercept", f"{stats['intercept']:.2f}")
        metric_cols[2].metric("R2", f"{stats['r2']:.3f}")
        st.caption(regression_interpretation(stats["slope"], stats["r2"], selected_label))

    st.subheader("Day-of-Week Sleep Patterns")
    st.pyplot(build_sleep_pattern_chart(sleep_view, user_choice), clear_figure=True)


st.set_page_config(page_title="Fitbit Analytics Dashboard", layout="wide")
st.title("Fitbit Analytics Dashboard")

min_date, max_date = get_date_bounds()
user_options = ["All Users"] + load_user_summary()["Id"].astype(int).astype(str).tolist()

st.sidebar.header("Filters")
page = st.sidebar.radio(
    "Page",
    [
        "Page 1: General Overview",
        "Page 2: Individual Summary",
        "Page 3: Sleep Analysis",
    ],
)
user_choice = st.sidebar.selectbox("User ID", user_options, index=0)
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if len(date_range) != 2:
    start_date, end_date = min_date, max_date
else:
    start_date, end_date = date_range
if start_date > end_date:
    start_date, end_date = end_date, start_date

st.caption(
    f"Sidebar filters apply to Pages 2-3. Current selection: {format_user_option(user_choice)}, "
    f"{start_date} to {end_date}."
)

if page == "Page 1: General Overview":
    render_page_one()
elif page == "Page 2: Individual Summary":
    render_page_two(user_choice, start_date, end_date)
else:
    render_page_three(user_choice, start_date, end_date)
