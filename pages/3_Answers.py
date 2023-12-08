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


st.title("5. Possible answers ")
st.subheader('5.1. Incresing number of Features')

st.write("Generally, increasing the number of features can improve a model's accuracy in predicting test data. Yet, in our scenario, we noted that augmenting the features led to only a marginal increase in accuracy, peaking at around 3 to 4%, depending on the specific model employed.")


data_table_1 = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'K-Nearest Neighbors', 'Naive Bayes'],
    'Accuracy': [0.562040, 0.460248, 0.507583, 0.567325, 0.487592, 0.555377],
    'Training Time': [62.889099, 66.748142, 2443.676233, 14839.603901, 15.620947, 15.620947],
    'Prediction Time': [0.000000, 0.000000, 93.730211, 4467.330933, 78.107119, 0.000000]
}

df_table_1 = pd.DataFrame(data_table_1)


data_table_2 = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'K-Nearest Neighbors', 'Naive Bayes'],
    'Accuracy': [0.572381, 0.463006, 0.554458, 0.566866, 0.497702, 0.554917],
    'Training Time': [188.268900, 156.213045, 3828.952789, 13341.201067, 31.241655, 0.000000],
    'Prediction Time': [0.000000, 0.000000, 93.730688, 4462.399483, 157.737494, 0.000000]
}

df_table_2 = pd.DataFrame(data_table_2)



# Display the data tables


# Select option for the user to choose metric
selected_metric = st.selectbox('Select Metrics', ['Accuracy', 'Training Time', 'Prediction Time'])

# Bar plot based on selected metric
st.subheader(f'Comparison: {selected_metric}')
fig3, ax = plt.subplots()

bar_width = 0.35
models = np.arange(len(df_table_1['Model']))

if selected_metric == 'Accuracy':
    rects1 = ax.bar(models - bar_width/2, df_table_1['Accuracy'], bar_width, label='2 Features')
    rects2 = ax.bar(models + bar_width/2, df_table_2['Accuracy'], bar_width, label='6 Features')
    ax.set_ylabel('Accuracy')
elif selected_metric == 'Training Time':
    rects1 = ax.bar(models - bar_width/2, df_table_1['Training Time'], bar_width, label='2 Features')
    rects2 = ax.bar(models + bar_width/2, df_table_2['Training Time'], bar_width, label='6 Features')
    ax.set_ylabel('Training Time (ms)')
else:  # Prediction Time
    rects1 = ax.bar(models - bar_width/2, df_table_1['Prediction Time'], bar_width, label='2 Features')
    rects2 = ax.bar(models + bar_width/2, df_table_2['Prediction Time'], bar_width, label='6 Features')
    ax.set_ylabel('Prediction Time (ms)')

ax.set_xticks(models)
ax.set_xticklabels(df_table_1['Model'], rotation=45, ha='right')
ax.legend(loc='lower right')

# Add values on top of each bar
for rect in rects1:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, f'{height:.3f}', ha='center', va='bottom', fontsize=6)

for rect in rects2:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, f'{height:.3f}', ha='center', va='bottom', fontsize=6)

st.pyplot(fig3)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
rs2=rs.drop(['date','city','country','home_score','away_score'],axis=1)


X= [rs2['HomeRank'],rs2['AwayRank']]
Y = rs2['Result']
X = rs2.iloc[:, 6:8]

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X=np.array(X)
# X=np.transpose(X)
#Y = label_encoder.fit_transform(rs2['Result'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=40)




# Create and train a k-Nearest Neighbors classifier
classifier=LogisticRegression()
classifier.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = classifier.predict(X_test)

# Set a probability threshold for label assignment
Y_pred_proba = classifier.predict_proba(X_test) 

X_samples = X[130:135]  # Displaying the first 5 samples for example
Y_pred_proba_samples = Y_pred_proba[130:135]
Y_test_samples = Y_test[130:135]

# Create a DataFrame
dataaa = {
    'Rank Home': X_samples[:, 0],
    'Rank Away': X_samples[:, 1],
    'Drawprob (0) ': [proba[0]*100 for proba in Y_pred_proba_samples],
    'Loseprob(1) ': [proba[1]*100 for proba in Y_pred_proba_samples],
    'winprob (2) ': [proba[2]*100 for proba in Y_pred_proba_samples],
    'True Result': Y_test_samples
}


st.subheader('5.2 Multi-label classification')
st.write('Multilabel classification is a type of classification task where each instance (or data point) can belong to multiple classes simultaneously. In other words, instead of assigning a single label to each instance, as in traditional classification tasks, multilabel classification allows instances to have multiple labels or categories associated with them.')
dffff= pd.DataFrame(dataaa)
st.table(dffff)
