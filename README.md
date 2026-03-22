# Fitbit Analytics Dashboard - Project group 9

This repository contains a Fitbit data analysis project built with Python, SQLite, and Streamlit. It combines daily activity, sleep, hourly steps, and intensity data into an interactive dashboard that helps explore both population-level trends and individual participant behavior.

The project focuses on turning raw Fitbit tracking data into clear insights about:

- physical activity patterns
- sleep behavior
- participant differences across the study period
- user classification based on average daily steps

## Features

### Interactive Dashboard

The Streamlit dashboard in [part5/a5.py](part5/a5.py) includes:

- **Page 1: General Overview**
  - KPI cards for total users, average daily steps, average daily sleep, and average daily calories
  - Trend chart for steps and calories over time
  - User class distribution donut chart
  - Stacked weekday activity-level distribution
  - Hour-by-day heatmap of average steps
  - Sortable per-user summary table with color gradients

- **Page 2: Individual Summary**
  - Personal KPI cards for user class, steps, sleep, and calories
  - Deltas compared to the study average
  - Percentile rankings for steps, sleep, and calories
  - Daily trend charts for the selected participant
  - Day-of-week comparison charts versus the study average
  - Hourly activity bar chart
  - Personal weekday activity distribution

- **Page 3: Sleep Analysis**
  - Regression panel with selectable explanatory variable
  - Scatter plot with regression line and confidence band
  - Slope, intercept, and R2 metrics
  - Automatic interpretation text
  - Day-of-week sleep pattern analysis

### Analytical Modules

The repository also includes supporting analysis scripts:

- [part1/a1.py](part1/a1.py): early exploratory plots and regression work
- [part3/a3.py](part3/a3.py): sleep, activity, weather, and time-block analysis
- [part4/a4.py](part4/a4.py): reusable helper functions and classification logic used by the dashboard

## Project Structure

```text
fitbit-project-9/
├── README.md
├── daily_activity.csv
├── fitbit_database.db
├── part1/
│   ├── a1.py
│   └── plots/
├── part3/
│   ├── a3.py
│   └── chicago 2016-03-12 to 2016-04-09.csv
│   └── plots/
├── part4/
│   └── a4.py
│   └──plots/
└── part5/
    └── a5.py
```

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- SQLite
- Streamlit

## Data Sources

The project uses:

- Fitbit activity and sleep data stored in `fitbit_database.db`
- supplementary CSV files such as `daily_activity.csv`
- weather data used in exploratory analysis

## User Classification

Participants are grouped into three classes based on average daily steps:

- **Light user**: fewer than 10,000 steps/day
- **Moderate user**: 10,000 to 14,999 steps/day
- **Heavy user**: 15,000+ steps/day

This classification logic is implemented in [part4/a4.py](part4/a4.py).

## How to Run

### 1. Install dependencies

Install the required packages in the working Python environment:

```bash
pip install streamlit pandas numpy matplotlib scikit-learn
```

### 2. Run the dashboard

From the repository root, start the Streamlit app with:

```bash
streamlit run part5/a5.py
```

## Project Goal

The goal of this project is to present Fitbit health and activity data in a way that is both analytically useful and easy to explore interactively. It is designed for comparing users, identifying activity and sleep patterns, and summarizing study-wide behavior through visual analytics.
