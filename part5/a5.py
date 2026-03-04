import streamlit as st
import pandas as pd
from datetime import date

st.set_page_config(
    page_title="Fitbit Analytics Dashboard",
    layout="wide"
)

st.title("📊 Fitbit Analytics Dashboard")
st.markdown("Mockup version — layout only (no database yet)")

st.sidebar.header("Filters")

# Mock user IDs
user_id = st.sidebar.selectbox(
    "Select User ID",
    [1503960366, 1624580081, 4020332650]
)

start_date = st.sidebar.date_input(
    "Start Date",
    date(2016, 4, 1)
)

end_date = st.sidebar.date_input(
    "End Date",
    date(2016, 4, 30)
)

time_block = st.sidebar.selectbox(
    "Time of Day",
    ["All Day", "0-4", "4-8", "8-12", "12-16", "16-20", "20-24"]
)

st.subheader("🔢 Key Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Steps", "105,234")
col2.metric("Avg Daily Steps", "8,456")
col3.metric("Total Calories", "12,543")
col4.metric("Avg Sleep (min)", "412")

tab1, tab2, tab3, tab4 = st.tabs([
    "Daily Activity",
    "Sleep Analysis",
    "Heart Rate",
    "4-Hour Blocks"
])

with tab1:
    st.subheader("Daily Steps Trend")
    st.line_chart(pd.DataFrame({
        "Steps": [5000, 7000, 8000, 6500, 9000]
    }))

with tab2:
    st.subheader("Sleep vs Active Minutes")
    st.scatter_chart(pd.DataFrame({
        "Active Minutes": [30, 45, 60, 20, 50],
        "Sleep Minutes": [420, 390, 410, 450, 400]
    }))

with tab3:
    st.subheader("Heart Rate Over Time")
    st.line_chart(pd.DataFrame({
        "Heart Rate": [70, 75, 80, 78, 72]
    }))

with tab4:
    st.subheader("Average Steps per 4-Hour Block")
    st.bar_chart(pd.DataFrame({
        "Steps": [200, 1500, 4000, 3500, 1800, 300]
    }, index=["0-4", "4-8", "8-12", "12-16", "16-20", "20-24"]))