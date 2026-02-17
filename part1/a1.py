# Only work with daily_activity.csv, for now.
# Things to do:
# 1. Basic inspection of data: 
#   - Count unique users
#   - Total distance for per user
#   - Display results in a graph
#   - Create function => Plot calories burnt per day
# 2. Relationship between Calories and Steps Taken
#   - Estimate calories burnt based on data
#   - Run linear regression
# 3. Check any additional features of dataset
#   - Variable relations => Create data visualization

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf


FILEPATH = "./daily_activity.csv"
PLOTPATH = "plots"
os.makedirs(PLOTPATH, exist_ok=True)
SHOWPLOTS = False

def showCaloriesPerDay(df, userId, timeRange=(None, None)):
    if not userId in df["Id"].values: return

    # sort data and filter columns
    userData = df[df["Id"] == userId][["ActivityDate", "Calories"]].copy()
    userData["ActivityDate"] = pd.to_datetime(userData["ActivityDate"])
    userData = userData.sort_values("ActivityDate")
    if not (timeRange[0] is None or timeRange[1] is None):
        userData = userData[(userData["ActivityDate"] >= timeRange[0]) & (userData["ActivityDate"] <= timeRange[1])]

    userData.plot(x="ActivityDate", y="Calories", kind="line", marker="o", figsize=(12, 6))
    plt.xlabel("Date")
    plt.ylabel("Calories Burnt")
    plt.title(f"Calories Burnt per Day for User {userId}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTPATH,  f"calories_per_day_user_{userId}.png"))
    if (SHOWPLOTS): plt.show()

def showWeekdayFrequency(df):
    df["ActivityDate"] = pd.to_datetime(df["ActivityDate"])
    df["Weekday"] = df["ActivityDate"].dt.day_name()
    weekdayFrequency = df["Weekday"].value_counts().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    
    weekdayFrequency.plot(kind="bar", figsize=(12, 6))
    plt.xlabel("Weekday")
    plt.ylabel("Frequency")
    plt.title("Frequency for Weekdays")
    plt.ylim(0, weekdayFrequency.max() * 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTPATH, "weekday_frequency.png"))

def basicInspection(df):
    # Unique users
    amountUsers = df["Id"].nunique()
    print("# unique users: ", amountUsers)

    # Total distance per user
    distancePerUser = df[["Id", "TotalDistance"]].groupby("Id")["TotalDistance"].sum().sort_values(ascending=True)
    
    # Results in a graph
    distancePerUser.plot(kind="bar", figsize=(12, 6))
    plt.xlabel("User Id")
    plt.ylabel("Total Distance")
    plt.title("Total Distance per User")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTPATH, "total_distance_per_user.png"))
    if (SHOWPLOTS): plt.show()

    # Calories burnt per day
    userId = 8877689391
    timeRange = (
        pd.Timestamp("2016-01-01"),
        pd.Timestamp("2016-12-31")
    )
    showCaloriesPerDay(df, userId, timeRange)

    # Plot frequency of runners for week days
    showWeekdayFrequency(df)

def relationshipCaloriesSteps(df):
    # Calories = b_0 + b_1 * TotalSteps + b_2 * Id + e
    model = smf.ols(formula="Calories ~ TotalSteps + C(Id)", data=df).fit() 
    return model

def indivdual_steps_vs_calories(df, id):
    
    user_data = df[df['Id'] == id]
    linear_model = relationshipCaloriesSteps(df)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(user_data['TotalSteps'], user_data['Calories'], alpha=0.7, label='Observed')

    user_data_sorted = user_data.sort_values('TotalSteps')
    calories_predictions = linear_model.predict(user_data_sorted)
    ax.plot(
        user_data_sorted['TotalSteps'],
        calories_predictions,
        color='red',
        linewidth=2,
        label='Regression line'
    )

    ax.set_xlabel('Total steps')
    ax.set_ylabel('Calories burnt')
    ax.set_title(f"Steps vs. Calories for User {id}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTPATH, f"steps_vs_calories_user_{id}.png"))
    if (SHOWPLOTS): plt.show()
    plt.close(fig)



def main():
    # Read in data
    df = pd.read_csv(FILEPATH)

    # Basic inspection of data
    basicInspection(df)

    # Relationship between calories and steps taken
    relationshipCaloriesSteps(df)

    # Plot for individual id
    indivdual_steps_vs_calories(df, 1503960366)

    # Additional data exploration
    ## Correlation matrix between the variables
    variables = df.drop(['Id', 'ActivityDate', 'Weekday'], axis = 1)
    correlations = variables.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlations, annot=True, cmap='coolwarm')
    plt.title('Correlations between the variables')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTPATH, f"correlation_matrix.png"))

if __name__ == "__main__":
    main()
