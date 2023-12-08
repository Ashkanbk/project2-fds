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
rs2=rs.drop(['date','city','country','home_score','away_score'],axis=1)



st.title("3. Classification models: ")
st.write("In the pursuit of predicting football match outcomes, a selection of machine learning models was employed for the classification and training of our dataset. The models, namely Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors, Naive Bayes, Gradient Boosting, and Neural Network, were utilized to discern patterns and relationships within the data. Each model was trained and evaluated, and their accuracy in predicting match results based on a comprehensive set of features, including team identities, match scores, and team rankings, was compared in this section.")




from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X= [rs2['HomeRank'],rs2['AwayRank']]
Y = rs2['Result']
X = rs2.iloc[:, 6:8]

label_encoder = LabelEncoder()
X=np.array(X)
#X=np.transpose(X)
#Y = label_encoder.fit_transform(rs2['Result'])
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=40)
# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Neural Network': MLPClassifier()
}

st.title('3.1 Model Implementation')

classifier_name = st.selectbox('Select Classifier', list(classifiers.keys()))
test_ratio = st.slider('Test Ratio', 0.1, 0.4, 0.2, 0.05)
random_seed = st.number_input('Random Seed', min_value=1, max_value=100, value=40, step=1)

# Train and evaluate the selected classifier
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio, random_state=random_seed)

classifier = classifiers[classifier_name]

start_time = time.time()  # Record the start time
classifier.fit(X_train, Y_train)
training_time = (time.time() - start_time) * 1000  # Convert to milliseconds

start_time = time.time()  # Record the start time for prediction
predictions = classifier.predict(X_test)
prediction_time = (time.time() - start_time) * 1000  # Convert to milliseconds

accuracy = accuracy_score(Y_test, predictions)

# Display results
st.subheader('Results')
results_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Training Time', 'Prediction Time'],
    'Value': [f'{accuracy:.2%}', f'{training_time:.2f} ms', f'{prediction_time:.2f} ms']
})
st.table(results_table)

# Display classification report
st.subheader('Classification Report')
classification_report_df = pd.DataFrame(classification_report(Y_test, predictions, output_dict=True)).T
st.table(classification_report_df)

# Display confusion matrix
st.subheader('Confusion Matrix')
conf_matrix = confusion_matrix(Y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens")
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot()


results = {'Model': [], 'Accuracy': [], 'Training Time': [], 'Prediction Time': []}

# Train and evaluate each classifier
for name, classifier in classifiers.items():
    start_time = time.time()  # Record the start time
    classifier.fit(X_train, Y_train)
    training_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    start_time = time.time()  # Record the start time for prediction
    predictions = classifier.predict(X_test)
    prediction_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    accuracy = accuracy_score(Y_test, predictions)

    results['Model'].append(name)
    results['Accuracy'].append(accuracy)
    results['Training Time'].append(training_time)
    results['Prediction Time'].append(prediction_time)
    
# Create a DataFrame from the results
df_results = pd.DataFrame(results)

# Streamlit App




# Select box for choosing a metric
st.title('3.2 Models Comparison')

# Provided data
datamodel = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'K-Nearest Neighbors', 'Naive Bayes', 'Gradient Boosting', 'Neural Network'],
    'Accuracy': [0.562040, 0.454733, 0.493566, 0.566176, 0.492188, 0.542969, 0.566866, 0.502068],
    'Training Time (ms)': [448.691130, 95.520258, 5524.929285, 54585.779428, 34.304857, 16.031742, 8615.590572, 2263.385773],
    'Prediction Time (ms)': [1.999855, 3.001928, 271.054745, 15281.101942, 453.765869, 2.737522, 60.298681, 3.998280]
}


# Create a DataFrame
df_resultsmodel = pd.DataFrame(datamodel)

# Create tabs
tabs = ["Accuracy", "Training Time", "Prediction Time"]
selected_tab = st.selectbox("Select a tab", tabs)

# Plot based on the selected tab
if selected_tab == "Accuracy":
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=df_resultsmodel)
    plt.xticks(rotation=45, ha='right')
    st.pyplot()

elif selected_tab == "Training Time":
    # Plot training time
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Training Time (ms)', data=df_resultsmodel)
    plt.xticks(rotation=45, ha='right')
    st.pyplot()

elif selected_tab == "Prediction Time":
    # Plot prediction time
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Prediction Time (ms)', data=df_resultsmodel)
    plt.xticks(rotation=45, ha='right')
    st.pyplot()


st.table(df_resultsmodel)