import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
DB_PATH = os.path.join(DATA_DIR, "fitbit_database.db")
WEATHER_PATH = os.path.join(DATA_DIR, "part3", "chicago 2016-03-12 to 2016-04-09.csv")

BLOCKS = ["0-4", "4-8", "8-12", "12-16", "16-20", "20-24"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CLASS_ORDER = ["Light user", "Moderate user", "Heavy user"]

# ── DB connection ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


# ── Helper functions ────────────────────────────────────────────────────────────
def assign_block(hour):
    if hour < 4:
        return "0-4"
    elif hour < 8:
        return "4-8"
    elif hour < 12:
        return "8-12"
    elif hour < 16:
        return "12-16"
    elif hour < 20:
        return "16-20"
    else:
        return "20-24"


def classify_ids(id_counts_df):
    rows = []
    for _, row in id_counts_df.iterrows():
        c = row["Count"]
        if c <= 10:
            cls = "Light user"
        elif c <= 15:
            cls = "Moderate user"
        else:
            cls = "Heavy user"
        rows.append({"Id": row["Id"], "Class": cls})
    df = pd.DataFrame(rows)
    df["Class"] = pd.Categorical(df["Class"], categories=CLASS_ORDER, ordered=True)
    return df.sort_values("Class").reset_index(drop=True)


# ── Cached data loaders ─────────────────────────────────────────────────────────
@st.cache_data
def load_daily_activity():
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT CAST(Id AS INTEGER) AS Id, ActivityDate, TotalSteps, TotalDistance, "
        "Calories, VeryActiveMinutes, FairlyActiveMinutes, LightlyActiveMinutes, "
        "SedentaryMinutes FROM daily_activity",
        conn,
    )
    df["ActivityDate"] = pd.to_datetime(df["ActivityDate"], format="mixed")
    df["date"] = df["ActivityDate"].dt.date
    df["weekday"] = df["ActivityDate"].dt.day_name()
    df["active_minutes"] = (
        df["VeryActiveMinutes"] + df["FairlyActiveMinutes"] + df["LightlyActiveMinutes"]
    )
    return df


@st.cache_data
def load_sleep():
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT CAST(Id AS INTEGER) AS Id, date, logId FROM minute_sleep", conn
    )
    df["datetime"] = pd.to_datetime(df["date"], format="mixed")
    sleep_logs = (
        df.groupby(["Id", "logId"])
        .agg(
            sleep_minutes=("date", "size"),
            date=("datetime", lambda s: s.dt.date.min()),
        )
        .reset_index()
    )
    sleep_logs["weekday"] = pd.to_datetime(sleep_logs["date"]).dt.day_name()
    sleep_logs["is_weekend"] = pd.to_datetime(sleep_logs["date"]).dt.weekday >= 5
    return sleep_logs


@st.cache_data
def load_hourly_steps():
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT CAST(Id AS INTEGER) AS Id, ActivityHour, StepTotal FROM hourly_steps", conn
    )
    df["ActivityHour"] = pd.to_datetime(df["ActivityHour"], format="mixed")
    df["date"] = df["ActivityHour"].dt.date
    df["hour"] = df["ActivityHour"].dt.hour
    df["block"] = df["hour"].apply(assign_block)
    return df


@st.cache_data
def load_hourly_calories():
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT CAST(Id AS INTEGER) AS Id, ActivityHour, Calories FROM hourly_calories", conn
    )
    df["ActivityHour"] = pd.to_datetime(df["ActivityHour"], format="mixed")
    df["date"] = df["ActivityHour"].dt.date
    df["hour"] = df["ActivityHour"].dt.hour
    df["block"] = df["hour"].apply(assign_block)
    return df


@st.cache_data
def load_hourly_intensity():
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT CAST(Id AS INTEGER) AS Id, ActivityHour, TotalIntensity FROM hourly_intensity",
        conn,
    )
    df["ActivityHour"] = pd.to_datetime(df["ActivityHour"], format="mixed")
    df["date"] = df["ActivityHour"].dt.date
    df["block"] = df["ActivityHour"].dt.hour.apply(assign_block)
    return df


@st.cache_data
def load_heart_rate():
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT CAST(Id AS INTEGER) AS Id, Time, Value FROM heart_rate", conn
    )
    df["Time"] = pd.to_datetime(df["Time"], format="mixed")
    df["date"] = df["Time"].dt.date
    return df


@st.cache_data
def load_weight():
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT CAST(Id AS INTEGER) AS Id, Date, WeightKg, BMI, Fat FROM weight_log", conn
    )
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    return df


@st.cache_data
def load_user_classes():
    df = load_daily_activity()
    id_counts = df.groupby("Id").size().reset_index(name="Count")
    return classify_ids(id_counts)


# ── Page helpers ────────────────────────────────────────────────────────────────
def fmt_num(n, decimals=0):
    if pd.isna(n):
        return "N/A"
    fmt = f"{{:,.{decimals}f}}"
    return fmt.format(n)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 – Overview
# ═══════════════════════════════════════════════════════════════════════════════
def page_overview(start_date, end_date):
    st.title("Fitbit Research – Overview")
    st.markdown(
        "General statistics for the Fitbit wearable study (March – April 2016, Chicago area)."
    )

    df_act = load_daily_activity()
    df_sleep = load_sleep()
    user_classes = load_user_classes()

    # Apply date filter
    mask = (df_act["date"] >= start_date) & (df_act["date"] <= end_date)
    df = df_act[mask].copy()

    sleep_mask = (df_sleep["date"] >= start_date) & (df_sleep["date"] <= end_date)
    df_sl = df_sleep[sleep_mask].copy()

    # ── Numerical summary ────────────────────────────────────────────────────
    st.subheader("Key Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Participants", fmt_num(df["Id"].nunique()))
    c2.metric("Avg Daily Steps", fmt_num(df["TotalSteps"].mean()))
    c3.metric("Avg Calories / Day", fmt_num(df["Calories"].mean()))
    c4.metric("Avg Active Min / Day", fmt_num(df["active_minutes"].mean()))
    avg_sleep = df_sl["sleep_minutes"].mean()
    c5.metric("Avg Sleep (min)", fmt_num(avg_sleep))

    st.divider()

    # ── Graphical summaries ──────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    # 1) Daily average steps over time
    with col_left:
        st.subheader("Daily Average Steps (all users)")
        daily_steps = df.groupby("date")["TotalSteps"].mean().reset_index()
        daily_steps.columns = ["Date", "Avg Steps"]
        st.line_chart(daily_steps.set_index("Date"), use_container_width=True)

    # 2) User class distribution
    with col_right:
        st.subheader("User Classification")
        class_counts = (
            user_classes["Class"].value_counts().reindex(CLASS_ORDER).fillna(0)
        )
        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ["#9ecae1", "#6baed6", "#3182bd"]
        ax.bar(class_counts.index, class_counts.values, color=colors)
        ax.set_ylabel("Number of Users")
        ax.set_title("Users by Activity Class")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Light user ≤ 10 days, Moderate user 11–15 days, Heavy user > 15 days of recorded activity."
        )

    st.divider()

    # 3) Day-of-week activity breakdown (stacked bar)
    st.subheader("Average Activity Breakdown by Day of Week")
    dow = (
        df.groupby("weekday")[
            ["SedentaryMinutes", "LightlyActiveMinutes", "FairlyActiveMinutes", "VeryActiveMinutes"]
        ]
        .mean()
        .reindex(DAY_ORDER)
    )
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    bottoms = np.zeros(len(dow))
    palette = ["#d9d9d9", "#a8ddb5", "#43a2ca", "#0868ac"]
    labels = ["Sedentary", "Lightly Active", "Fairly Active", "Very Active"]
    for col, color, lbl in zip(
        ["SedentaryMinutes", "LightlyActiveMinutes", "FairlyActiveMinutes", "VeryActiveMinutes"],
        palette,
        labels,
    ):
        ax2.bar(dow.index, dow[col].values, bottom=bottoms, color=color, label=lbl)
        bottoms += dow[col].fillna(0).values
    ax2.set_ylabel("Average Minutes")
    ax2.set_title("Activity Composition per Day of Week")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.divider()

    # 4) Numerical summary table
    st.subheader("Summary Statistics Table")
    summary = (
        df[["TotalSteps", "Calories", "active_minutes", "SedentaryMinutes", "VeryActiveMinutes"]]
        .describe()
        .T.rename(
            index={
                "TotalSteps": "Total Steps",
                "Calories": "Calories",
                "active_minutes": "Active Minutes",
                "SedentaryMinutes": "Sedentary Minutes",
                "VeryActiveMinutes": "Very Active Minutes",
            }
        )
    )
    st.dataframe(summary.style.format("{:.1f}"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 – Individual Profile
# ═══════════════════════════════════════════════════════════════════════════════
def page_individual(user_id, start_date, end_date, time_block):
    st.title(f"Individual Profile – User {user_id}")

    df_act = load_daily_activity()
    df_sleep = load_sleep()
    df_hr = load_heart_rate()
    df_intensity = load_hourly_intensity()
    user_classes = load_user_classes()

    # Filter by user & date
    u_act = df_act[
        (df_act["Id"] == user_id)
        & (df_act["date"] >= start_date)
        & (df_act["date"] <= end_date)
    ].copy()

    u_sleep = df_sleep[
        (df_sleep["Id"] == user_id)
        & (df_sleep["date"] >= start_date)
        & (df_sleep["date"] <= end_date)
    ].copy()

    u_hr = df_hr[
        (df_hr["Id"] == user_id)
        & (df_hr["date"] >= start_date)
        & (df_hr["date"] <= end_date)
    ].copy()

    u_intensity = df_intensity[
        (df_intensity["Id"] == user_id)
        & (df_intensity["date"] >= start_date)
        & (df_intensity["date"] <= end_date)
    ].copy()

    # Apply time-of-day block filter
    if time_block != "All Day":
        u_hr_block = u_hr[u_hr["Time"].dt.hour.apply(assign_block) == time_block]
        u_intensity_block = u_intensity[u_intensity["block"] == time_block]
    else:
        u_hr_block = u_hr
        u_intensity_block = u_intensity

    # User class
    cls_row = user_classes[user_classes["Id"] == user_id]
    user_class = cls_row["Class"].values[0] if len(cls_row) else "Unknown"

    st.markdown(f"**Activity Class:** {user_class}")

    # ── Key metrics ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Steps", fmt_num(u_act["TotalSteps"].sum()))
    c2.metric("Avg Daily Steps", fmt_num(u_act["TotalSteps"].mean()))
    c3.metric("Avg Calories / Day", fmt_num(u_act["Calories"].mean()))
    avg_sleep = u_sleep["sleep_minutes"].mean()
    c4.metric("Avg Sleep (min)", fmt_num(avg_sleep))

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Daily Activity", "Sleep Patterns", "Heart Rate & Intensity", "4-Hour Blocks"]
    )

    # ── Tab 1: Daily activity ────────────────────────────────────────────────
    with tab1:
        if u_act.empty:
            st.info("No activity data for the selected period.")
        else:
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("Steps & Calories over Time")
                chart_df = u_act.set_index("ActivityDate")[["TotalSteps", "Calories"]]
                st.line_chart(chart_df, use_container_width=True)

            with col_r:
                st.subheader("Avg Steps & Calories by Day of Week")
                dow = (
                    u_act.groupby("weekday")[["TotalSteps", "Calories"]]
                    .mean()
                    .reindex(DAY_ORDER)
                    .dropna(how="all")
                )
                fig, ax = plt.subplots(figsize=(6, 3.5))
                x = np.arange(len(dow))
                w = 0.4
                ax.bar(x - w / 2, dow["TotalSteps"].values, w, label="Steps", color="steelblue")
                ax.bar(x + w / 2, dow["Calories"].values, w, label="Calories", color="tomato")
                ax.set_xticks(x)
                ax.set_xticklabels(dow.index, rotation=30, ha="right")
                ax.legend(fontsize=8)
                ax.grid(axis="y", linestyle="--", alpha=0.4)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # ── Tab 2: Sleep ─────────────────────────────────────────────────────────
    with tab2:
        if u_sleep.empty:
            st.info("No sleep data for the selected period.")
        else:
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("Sleep Duration over Time")
                sl_time = u_sleep.sort_values("date")
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.plot(
                    pd.to_datetime(sl_time["date"]),
                    sl_time["sleep_minutes"],
                    marker="o",
                    linewidth=1.2,
                    color="mediumseagreen",
                )
                ax.axhline(7 * 60, linestyle="--", color="grey", alpha=0.5, label="7 h")
                ax.set_ylabel("Sleep (min)")
                ax.legend(fontsize=8)
                ax.grid(linestyle="--", alpha=0.4)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with col_r:
                st.subheader("Sleep by Day of Week")
                box_data, labels = [], []
                for d in DAY_ORDER:
                    vals = u_sleep.loc[u_sleep["weekday"] == d, "sleep_minutes"].dropna().values
                    if len(vals) > 0:
                        box_data.append(vals)
                        labels.append(d)
                if box_data:
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    ax.boxplot(box_data, tick_labels=labels, showfliers=False)
                    ax.axhline(420, linestyle="--", color="grey", alpha=0.5, label="7 h")
                    ax.set_ylabel("Sleep (min)")
                    ax.tick_params(axis="x", rotation=30)
                    ax.grid(axis="y", linestyle="--", alpha=0.4)
                    ax.legend(fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Not enough data for box plot.")

    # ── Tab 3: Heart rate & intensity ────────────────────────────────────────
    with tab3:
        st.caption(f"Showing time block: **{time_block}**")
        if u_hr_block.empty and u_intensity_block.empty:
            st.info("No heart rate or intensity data for the selected period / time block.")
        else:
            if not u_hr_block.empty:
                st.subheader("Heart Rate over Time")
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.plot(
                    u_hr_block["Time"],
                    u_hr_block["Value"],
                    linewidth=0.6,
                    color="crimson",
                )
                ax.set_ylabel("Heart Rate (bpm)")
                ax.grid(linestyle="--", alpha=0.4)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            if not u_intensity_block.empty:
                st.subheader("Exercise Intensity (hourly)")
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.bar(
                    u_intensity_block["ActivityHour"],
                    u_intensity_block["TotalIntensity"],
                    width=1 / 24,
                    color="steelblue",
                )
                ax.set_ylabel("Total Intensity")
                ax.grid(axis="y", linestyle="--", alpha=0.4)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # ── Tab 4: 4-hour blocks ─────────────────────────────────────────────────
    with tab4:
        df_steps = load_hourly_steps()
        df_cals = load_hourly_calories()

        u_steps = df_steps[
            (df_steps["Id"] == user_id)
            & (df_steps["date"] >= start_date)
            & (df_steps["date"] <= end_date)
        ]
        u_cals = df_cals[
            (df_cals["Id"] == user_id)
            & (df_cals["date"] >= start_date)
            & (df_cals["date"] <= end_date)
        ]

        avg_steps = (
            u_steps.groupby(["date", "block"])["StepTotal"]
            .sum()
            .groupby("block")
            .mean()
            .reindex(BLOCKS)
        )
        avg_cals = (
            u_cals.groupby(["date", "block"])["Calories"]
            .sum()
            .groupby("block")
            .mean()
            .reindex(BLOCKS)
        )
        avg_sleep_block = None
        if not u_sleep.empty:
            df_min_sleep_raw = pd.read_sql_query(
                "SELECT CAST(Id AS INTEGER) AS Id, date, logId FROM minute_sleep WHERE CAST(Id AS INTEGER) = ?",
                get_conn(),
                params=(user_id,),
            )
            df_min_sleep_raw["datetime"] = pd.to_datetime(df_min_sleep_raw["date"], format="mixed")
            df_min_sleep_raw["date_only"] = df_min_sleep_raw["datetime"].dt.date
            df_min_sleep_raw["block"] = df_min_sleep_raw["datetime"].dt.hour.apply(assign_block)
            date_mask = (df_min_sleep_raw["date_only"] >= start_date) & (
                df_min_sleep_raw["date_only"] <= end_date
            )
            df_min_sleep_raw = df_min_sleep_raw[date_mask]
            if not df_min_sleep_raw.empty:
                avg_sleep_block = (
                    df_min_sleep_raw.groupby(["logId", "block"])
                    .size()
                    .reset_index(name="sleep_min")
                    .groupby("block")["sleep_min"]
                    .mean()
                    .reindex(BLOCKS)
                )

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].bar(BLOCKS, avg_steps.values, color="steelblue")
        axes[0].set_title("Avg Steps per 4-Hour Block")
        axes[0].set_ylabel("Steps")
        axes[0].grid(axis="y", linestyle="--", alpha=0.4)

        axes[1].bar(BLOCKS, avg_cals.values, color="tomato")
        axes[1].set_title("Avg Calories per 4-Hour Block")
        axes[1].set_ylabel("Calories")
        axes[1].grid(axis="y", linestyle="--", alpha=0.4)

        if avg_sleep_block is not None:
            axes[2].bar(BLOCKS, avg_sleep_block.values, color="mediumseagreen")
            axes[2].set_title("Avg Sleep Min per 4-Hour Block")
            axes[2].set_ylabel("Minutes")
            axes[2].grid(axis="y", linestyle="--", alpha=0.4)
        else:
            axes[2].text(0.5, 0.5, "No sleep data", ha="center", va="center", transform=axes[2].transAxes)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 – Sleep Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def page_sleep(start_date, end_date):
    st.title("Sleep Duration Analysis")
    st.markdown(
        "Explore which variables affect how long individuals sleep. "
        "Data merged from `daily_activity` and `minute_sleep`."
    )

    df_act = load_daily_activity()
    df_sleep = load_sleep()
    user_classes = load_user_classes()

    # Filter dates
    act_mask = (df_act["date"] >= start_date) & (df_act["date"] <= end_date)
    slp_mask = (df_sleep["date"] >= start_date) & (df_sleep["date"] <= end_date)
    df_a = df_act[act_mask].copy()
    df_s = df_sleep[slp_mask].copy()

    merged = pd.merge(df_a, df_s[["Id", "date", "sleep_minutes", "is_weekend"]], on=["Id", "date"], how="inner")
    merged = merged.merge(user_classes, on="Id", how="left")

    if merged.empty:
        st.warning("No data available for the selected date range.")
        return

    # ── Section 1: Sleep vs Active Minutes (regression) ──────────────────────
    st.subheader("1. Sleep Duration vs Active Minutes")
    col_l, col_r = st.columns([2, 1])

    with col_l:
        X = merged[["active_minutes"]].values
        y = merged["sleep_minutes"].values
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_

        x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        y_line = model.predict(x_line)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(X, y, alpha=0.4, s=15, color="steelblue", label="Observations")
        ax.plot(x_line, y_line, color="crimson", linewidth=2, label=f"Regression (R²={r2:.3f})")
        ax.set_xlabel("Active Minutes")
        ax.set_ylabel("Sleep Minutes")
        ax.set_title("Sleep vs Active Minutes")
        ax.legend(fontsize=8)
        ax.grid(linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_r:
        st.markdown("**Regression results**")
        st.metric("Slope", f"{slope:.3f} min/active-min")
        st.metric("Intercept", f"{intercept:.1f} min")
        st.metric("R²", f"{r2:.4f}")
        corr = merged["active_minutes"].corr(merged["sleep_minutes"])
        st.metric("Pearson r", f"{corr:.3f}")
        st.markdown(
            "A negative slope indicates that more active minutes are associated with "
            "slightly shorter (or equal) sleep durations."
        )

    st.divider()

    # ── Section 2: Sleep vs Steps ────────────────────────────────────────────
    st.subheader("2. Sleep Duration vs Total Steps")
    col_l2, col_r2 = st.columns([2, 1])
    with col_l2:
        X2 = merged[["TotalSteps"]].values
        y2 = merged["sleep_minutes"].values
        model2 = LinearRegression().fit(X2, y2)
        r2_2 = model2.score(X2, y2)

        x_line2 = np.linspace(X2.min(), X2.max(), 200).reshape(-1, 1)
        y_line2 = model2.predict(x_line2)

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.scatter(X2, y2, alpha=0.4, s=15, color="darkorange", label="Observations")
        ax2.plot(x_line2, y_line2, color="navy", linewidth=2, label=f"R²={r2_2:.3f}")
        ax2.set_xlabel("Total Steps")
        ax2.set_ylabel("Sleep Minutes")
        ax2.set_title("Sleep vs Total Steps")
        ax2.legend(fontsize=8)
        ax2.grid(linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with col_r2:
        st.markdown("**Regression results**")
        st.metric("Slope", f"{model2.coef_[0]:.5f} min/step")
        st.metric("R²", f"{r2_2:.4f}")
        corr2 = merged["TotalSteps"].corr(merged["sleep_minutes"])
        st.metric("Pearson r", f"{corr2:.3f}")

    st.divider()

    # ── Section 3: Sleep by User Class ───────────────────────────────────────
    st.subheader("3. Sleep Duration by User Class")
    col_l3, col_r3 = st.columns(2)

    with col_l3:
        box_data = [
            merged.loc[merged["Class"] == c, "sleep_minutes"].dropna().values
            for c in CLASS_ORDER
        ]
        counts = [len(d) for d in box_data]
        labels_cls = [f"{c}\n(n={n})" for c, n in zip(CLASS_ORDER, counts)]
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.boxplot([d for d in box_data if len(d) > 0],
                    tick_labels=[l for l, d in zip(labels_cls, box_data) if len(d) > 0],
                    showfliers=False)
        ax3.axhline(420, linestyle="--", color="grey", alpha=0.5, label="7 h")
        ax3.set_ylabel("Sleep (min)")
        ax3.set_title("Sleep by Activity Class")
        ax3.grid(axis="y", linestyle="--", alpha=0.4)
        ax3.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with col_r3:
        class_sleep = (
            merged.groupby("Class")["sleep_minutes"]
            .agg(["mean", "median", "std", "count"])
            .reindex(CLASS_ORDER)
            .rename(columns={"mean": "Mean", "median": "Median", "std": "Std Dev", "count": "N"})
        )
        st.markdown("**Mean sleep per class**")
        st.dataframe(class_sleep.style.format({"Mean": "{:.1f}", "Median": "{:.1f}", "Std Dev": "{:.1f}", "N": "{:.0f}"}))

    st.divider()

    # ── Section 4: Weekend vs Weekday ────────────────────────────────────────
    st.subheader("4. Weekend vs Weekday Sleep")
    col_l4, col_r4 = st.columns(2)

    with col_l4:
        wknd = merged.groupby("is_weekend")["sleep_minutes"].agg(["mean", "median", "std", "count"])
        wknd.index = ["Weekday", "Weekend"]
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        bars = ax4.bar(wknd.index, wknd["mean"], color=["steelblue", "tomato"], width=0.4)
        ax4.errorbar(wknd.index, wknd["mean"], yerr=wknd["std"], fmt="none", color="black", capsize=5)
        ax4.axhline(420, linestyle="--", color="grey", alpha=0.5)
        ax4.set_ylabel("Mean Sleep (min)")
        ax4.set_title("Sleep: Weekend vs Weekday")
        ax4.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    with col_r4:
        st.markdown("**Descriptive statistics**")
        st.dataframe(wknd.rename(columns={"mean": "Mean", "median": "Median", "std": "Std Dev", "count": "N"})
                     .style.format({"Mean": "{:.1f}", "Median": "{:.1f}", "Std Dev": "{:.1f}", "N": "{:.0f}"}))
        diff = wknd.loc["Weekend", "mean"] - wknd.loc["Weekday", "mean"]
        st.markdown(f"Weekend sleep is **{diff:+.1f} min** compared to weekday sleep.")

    st.divider()

    # ── Section 5: Sleep by day of week ─────────────────────────────────────
    st.subheader("5. Sleep Duration by Day of Week")
    dow_sleep = merged.groupby("weekday")["sleep_minutes"].agg(["mean", "std"]).reindex(DAY_ORDER).dropna()
    fig5, ax5 = plt.subplots(figsize=(10, 3.5))
    ax5.bar(dow_sleep.index, dow_sleep["mean"], color="mediumseagreen", width=0.5, label="Mean Sleep")
    ax5.errorbar(dow_sleep.index, dow_sleep["mean"], yerr=dow_sleep["std"], fmt="none", color="black", capsize=5)
    ax5.axhline(420, linestyle="--", color="grey", alpha=0.5, label="7 h target")
    ax5.set_ylabel("Mean Sleep (min)")
    ax5.set_title("Average Sleep Duration by Day of Week")
    ax5.legend(fontsize=8)
    ax5.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

    st.divider()

    # ── Section 6: Sedentary minutes vs sleep ───────────────────────────────
    st.subheader("6. Sedentary Time vs Sleep Duration")
    col_l6, col_r6 = st.columns([2, 1])
    with col_l6:
        X6 = merged[["SedentaryMinutes"]].values
        y6 = merged["sleep_minutes"].values
        model6 = LinearRegression().fit(X6, y6)
        r2_6 = model6.score(X6, y6)

        x_line6 = np.linspace(X6.min(), X6.max(), 200).reshape(-1, 1)
        y_line6 = model6.predict(x_line6)

        fig6, ax6 = plt.subplots(figsize=(7, 4))
        ax6.scatter(X6, y6, alpha=0.4, s=15, color="plum", label="Observations")
        ax6.plot(x_line6, y_line6, color="darkviolet", linewidth=2, label=f"R²={r2_6:.3f}")
        ax6.set_xlabel("Sedentary Minutes")
        ax6.set_ylabel("Sleep Minutes")
        ax6.set_title("Sleep vs Sedentary Minutes")
        ax6.legend(fontsize=8)
        ax6.grid(linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

    with col_r6:
        corr6 = merged["SedentaryMinutes"].corr(merged["sleep_minutes"])
        st.metric("R²", f"{r2_6:.4f}")
        st.metric("Pearson r", f"{corr6:.3f}")
        st.markdown("More sedentary time may indicate less physical exertion, potentially affecting sleep quality and duration.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 – Activity Blocks (population-level)
# ═══════════════════════════════════════════════════════════════════════════════
def page_activity_blocks(start_date, end_date, time_block):
    st.title("Activity in 4-Hour Time Blocks")
    st.markdown("Population-level averages across all users for the selected date range.")

    df_steps = load_hourly_steps()
    df_cals = load_hourly_calories()
    df_sleep_raw = pd.read_sql_query(
        "SELECT CAST(Id AS INTEGER) AS Id, date, logId FROM minute_sleep", get_conn()
    )
    df_sleep_raw["datetime"] = pd.to_datetime(df_sleep_raw["date"], format="mixed")
    df_sleep_raw["date_only"] = df_sleep_raw["datetime"].dt.date
    df_sleep_raw["block"] = df_sleep_raw["datetime"].dt.hour.apply(assign_block)

    # Date filter
    steps_m = (df_steps["date"] >= start_date) & (df_steps["date"] <= end_date)
    cals_m = (df_cals["date"] >= start_date) & (df_cals["date"] <= end_date)
    slp_m = (df_sleep_raw["date_only"] >= start_date) & (df_sleep_raw["date_only"] <= end_date)

    df_s = df_steps[steps_m]
    df_c = df_cals[cals_m]
    df_sl = df_sleep_raw[slp_m]

    blocks_to_show = [time_block] if time_block != "All Day" else BLOCKS

    avg_steps = (
        df_s.groupby(["Id", "date", "block"])["StepTotal"]
        .sum()
        .groupby("block")
        .mean()
        .reindex(blocks_to_show)
    )
    avg_cals = (
        df_c.groupby(["Id", "date", "block"])["Calories"]
        .sum()
        .groupby("block")
        .mean()
        .reindex(blocks_to_show)
    )
    avg_sleep = (
        df_sl.groupby(["Id", "logId", "block"])
        .size()
        .reset_index(name="sleep_min")
        .groupby("block")["sleep_min"]
        .mean()
        .reindex(blocks_to_show)
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(blocks_to_show, avg_steps.values, color="steelblue")
    axes[0].set_title("Avg Steps per Block")
    axes[0].set_xlabel("Block")
    axes[0].set_ylabel("Steps")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    axes[1].bar(blocks_to_show, avg_cals.values, color="tomato")
    axes[1].set_title("Avg Calories per Block")
    axes[1].set_xlabel("Block")
    axes[1].set_ylabel("Calories")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    axes[2].bar(blocks_to_show, avg_sleep.values, color="mediumseagreen")
    axes[2].set_title("Avg Sleep Min per Block")
    axes[2].set_xlabel("Block")
    axes[2].set_ylabel("Minutes")
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Class breakdown
    st.subheader("Average Steps per Block by User Class")
    user_classes = load_user_classes()
    df_s_cls = df_steps[steps_m].merge(user_classes, on="Id", how="inner")
    class_block = (
        df_s_cls.groupby(["Class", "Id", "date", "block"], as_index=False)["StepTotal"]
        .sum()
        .groupby(["Class", "block"], as_index=False)["StepTotal"]
        .mean()
        .pivot(index="Class", columns="block", values="StepTotal")
        .reindex(index=CLASS_ORDER, columns=BLOCKS)
    )

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(BLOCKS))
    w = 0.25
    colors = ["#9ecae1", "#6baed6", "#08519c"]
    for i, cls in enumerate(CLASS_ORDER):
        vals = class_block.loc[cls].reindex(BLOCKS).values if cls in class_block.index else np.zeros(len(BLOCKS))
        ax2.bar(x + (i - 1) * w, vals, w, label=cls, color=colors[i])
    ax2.set_xticks(x)
    ax2.set_xticklabels(BLOCKS)
    ax2.set_ylabel("Avg Steps")
    ax2.set_title("Intra-Day Steps by User Class")
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 – Weight & BMI
# ═══════════════════════════════════════════════════════════════════════════════
def page_weight():
    st.title("Weight & BMI")
    st.markdown("Logged weight measurements for users who used the weight tracking feature.")

    df = load_weight()
    user_classes = load_user_classes()

    if df.empty:
        st.warning("No weight data available.")
        return

    df = df.merge(user_classes, on="Id", how="left")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("BMI Distribution by User Class")
        box_data = [
            df.loc[df["Class"] == c, "BMI"].dropna().values for c in CLASS_ORDER
        ]
        labels_cls = [
            f"{c} (n={len(d)})" for c, d in zip(CLASS_ORDER, box_data)
        ]
        fig, ax = plt.subplots(figsize=(6, 4))
        non_empty = [(d, l) for d, l in zip(box_data, labels_cls) if len(d) > 0]
        if non_empty:
            ax.boxplot([d for d, _ in non_empty], tick_labels=[l for _, l in non_empty], showfliers=False)
        ax.set_ylabel("BMI")
        ax.set_title("BMI by Activity Class")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_r:
        st.subheader("BMI vs Average Steps")
        # Merge with activity
        df_act = load_daily_activity()
        avg_steps = df_act.groupby("Id")["TotalSteps"].mean().reset_index()
        avg_steps.columns = ["Id", "AvgSteps"]
        df_bmi = df.groupby("Id")["BMI"].mean().reset_index()
        bmi_steps = df_bmi.merge(avg_steps, on="Id")

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(bmi_steps["AvgSteps"], bmi_steps["BMI"], alpha=0.7, color="darkorange")
        ax2.set_xlabel("Avg Daily Steps")
        ax2.set_ylabel("BMI")
        ax2.set_title("BMI vs Avg Daily Steps (per user)")
        ax2.grid(linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.subheader("Weight Trend over Time (per user)")
    weight_users = sorted(df["Id"].unique())
    sel = st.selectbox("Select User", weight_users, key="weight_user_sel")
    u_df = df[df["Id"] == sel].sort_values("Date")
    if u_df.empty:
        st.info("No weight data for this user.")
    else:
        fig3, ax3 = plt.subplots(figsize=(10, 3.5))
        ax3.plot(u_df["Date"], u_df["WeightKg"], marker="o", color="steelblue", label="Weight (kg)")
        ax_bmi = ax3.twinx()
        ax_bmi.plot(u_df["Date"], u_df["BMI"], marker="s", color="tomato", linestyle="--", label="BMI")
        ax3.set_ylabel("Weight (kg)")
        ax_bmi.set_ylabel("BMI", color="tomato")
        ax3.set_title(f"User {sel}: Weight & BMI over Time")
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax_bmi.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
        ax3.grid(linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="Fitbit Analytics Dashboard", layout="wide")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.title("📊 Fitbit Analytics Dashboard")
    st.sidebar.divider()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Individual Profile", "Sleep Analysis", "Activity Blocks", "Weight & BMI"],
    )
    st.sidebar.divider()
    st.sidebar.header("Filters")

    # Date range
    import datetime
    min_date = datetime.date(2016, 3, 12)
    max_date = datetime.date(2016, 4, 9)
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    if start_date > end_date:
        st.sidebar.error("Start date must be before end date.")
        return

    # Time block
    time_block = st.sidebar.selectbox(
        "Time of Day",
        ["All Day"] + BLOCKS,
    )

    # User ID (only relevant for Individual Profile)
    df_act = load_daily_activity()
    all_ids = sorted(df_act["Id"].unique().tolist())
    user_id = st.sidebar.selectbox("User ID (Individual Profile)", all_ids)

    st.sidebar.divider()
    st.sidebar.caption("Fitbit Wearable Study · March–April 2016")

    # ── Route to page ─────────────────────────────────────────────────────────
    if page == "Overview":
        page_overview(start_date, end_date)
    elif page == "Individual Profile":
        page_individual(user_id, start_date, end_date, time_block)
    elif page == "Sleep Analysis":
        page_sleep(start_date, end_date)
    elif page == "Activity Blocks":
        page_activity_blocks(start_date, end_date, time_block)
    elif page == "Weight & BMI":
        page_weight()


if __name__ == "__main__":
    main()
