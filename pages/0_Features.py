import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
import random

st.set_option('deprecation.showPyplotGlobalUse', False)



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

st.title("2. Feature selection:")


# Key Features
rrrr = pd.DataFrame(data)
rrrr = rrrr.reset_index(drop=True)
# Display the table in Streamlit
st.dataframe(rrrr)

st.subheader("Team Rankings:")
st.write("- Importance: High\n- Features: 'HomeRank', 'AwayRank', 'HomRankDiff'\n- Explanation: Significantly important, offering insights into relative team strengths.")

st.subheader("Tournament Type:")
st.write("- Importance: Medium\n- Feature: 'tournament'\n- Explanation: Provides moderate impact, capturing strategic variations based on tournament types.")

st.subheader("Venue:")
st.write("- Importance: Medium\n- Features: 'neutral'\n- Explanation: Binary indicator for venue neutrality, capturing potential home-field advantages.")

st.subheader("Date-Related Features:")
st.write("- Importance: Low to Medium\n- Features: 'date' and 'rank_date'\n- Explanation: Contribute to a lesser extent, incorporating temporal and ranking dynamics.")

st.write('First I started with 2 features of team rankings for classification.')