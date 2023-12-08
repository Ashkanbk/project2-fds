import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
import random

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Prediction of Football Matches")

st.image("FFF.jpg", use_column_width="always")

ra=pd.read_csv('fifa.csv')
rs=pd.read_csv('results.csv')

def determine_result(row):
    if row['home_score'] > row['away_score']:
        return 'Win'
    elif row['home_score'] < row['away_score']:
        return 'Loss'
    else:
        return 'Draw'
# Apply the custom function to create the 'Result' column
rs['Result'] = rs.apply(determine_result, axis=1)


radat = ra['rank_date'].unique()
rada = pd.DataFrame({'rank_date': radat})
rs['date'] = pd.to_datetime(rs['date'])
rada['rank_date'] = pd.to_datetime(rada['rank_date'])
ra['rank_date'] = pd.to_datetime(ra['rank_date'])

rada.sort_values(by='rank_date', inplace=True)
rs = pd.merge_asof(rs, rada, left_on='date', right_on='rank_date', direction='backward')
dictt = ra.set_index(['country_full', 'rank_date'])['rank'].to_dict()

rs = rs.dropna(subset=['rank_date'])


def get_rank(row, home_or_away):
    country = row[home_or_away + '_team']
    date = row['rank_date']
    
    return dictt.get((country, pd.to_datetime(date)),'NaN')

# Add columns for rank of the home and away teams
rs['HomeRank'] = rs.apply(get_rank, args=('home',), axis=1)
rs['AwayRank'] = rs.apply(get_rank, args=('away',), axis=1)

rs['HomeRank'] = pd.to_numeric(rs['HomeRank'], errors='coerce')
rs['AwayRank'] = pd.to_numeric(rs['AwayRank'], errors='coerce')

# Handle missing values (e.g., dropping rows with NaN values)
rs.dropna(subset=['HomeRank', 'AwayRank'], inplace=True)

rs['HomRankDiff']=rs['AwayRank']-rs['HomeRank']
data=rs









st.title("1. Introduction ")
st.write("The **FIFA ranking** system is a pivotal index in international football, assessing team strength through a point-based algorithm. Teams earn or lose points based on match results and opponent strength. These rankings, announced monthly, offer a numerical representation of a team's global standing, with **higher** rankings indicating **stronger** teams. They influence key factors like tournament seeding, shaping the competitive landscape of international football.")

rrrr = pd.DataFrame(data)
rrrr = rrrr.reset_index(drop=True)
# Display the table in Streamlit
st.dataframe(rrrr)



st.write("In this specific scatter plot, away teams are plotted against home teams, and the teams with lower-ranking numbers are considered stronger. The plot features a y=x line that divides the domain into two distinct regions. Points in the upper part of the plot indicate that a team playing at home is competing against a weaker team, and conversely in the lower part. When data points cluster closer to the y=x dashed line, it signifies that the teams are more evenly matched. To further dissect the results, the data is categorized by the outcome of the host team, whether they lost, won, or drew. Given the extensive number of data points, a random sample is plotted for better visualization, and the plot's size is adjustable")

# Main content with filters
result = st.multiselect("Result", data['Result'].unique())
filtered_data=data
if result:
    filtered_data = data[data['Result'].isin(result)]


sample_size = 1000


filtered_data = filtered_data.sample(sample_size,random_state=2)

# Create and display the scatter plot
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 8))
x = [1, 200]
y = [1, 200]

cupa = {'Win': 'seagreen', 'Loss': 'tomato', 'Draw': 'dimgrey'}

sns.scatterplot(data=filtered_data, x="HomeRank", y="AwayRank", palette=cupa, hue="Result", ax=ax)
plt.plot(x, y, 'k--', linewidth=4)


ax.set_xlabel("Home Rank")
ax.set_ylabel("Away Rank")
ax.set_title("Scatter Plot")
ax.legend(title="Result")
ax.set_aspect('equal')

st.pyplot(fig)





