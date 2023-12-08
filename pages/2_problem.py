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

st.title("4. Why this happened? ")
st.write("The accuracy of our model in predicting football match outcomes hovers around 0.6, a result influenced by several factors. Firstly, the limitations of our database and computational resources constrain the model's ability to capture all relevant aspects influencing match results. Football matches are inherently dynamic, and numerous unpredictable elements come into play that are not easily reducible to finite features. For instance, the impact of a last-minute player substitution, unexpected weather conditions, or an unanticipated change in team strategy can significantly alter the course of a match. Moreover, the nature of sports prediction suggests that achieving accuracy between 60 to 70% is considered quite respectable. Sports events inherently possess an element of unpredictability, making it challenging to achieve higher accuracy levels. In this context, our model's performance within the 60% range aligns with the expectations for predicting the often unpredictable outcomes of football matches.")

st.subheader('Decision Boundaries')


from matplotlib.colors import ListedColormap

st.image("download.png", use_column_width="always")

