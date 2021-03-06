import pandas as pd

runs = pd.read_csv('data/runs.csv')
runs.head()

races = pd.read_csv('data/races.csv')
races.head()

#Select features for modeling
runs_data = runs[['race_id', 'won', 'horse_age', 'horse_country', 'horse_type', 'horse_rating','horse_gear', 'declared_weight', 'actual_weight', 'draw', 'win_odds','place_odds', 'horse_id']]
runs_data.head()

races_data = races[['race_id', 'venue', 'config', 'surface', 'distance', 'going', 'race_class', 'date']]
races_data.head()

# merge the two datasets based on race_id column
df = pd.merge(runs_data, races_data)
df.head()

#Check missing values
df.isnull().any()

df.horse_country.isnull().value_counts(ascending=True)
df.horse_type.isnull().value_counts(ascending=True)
df.place_odds.isnull().value_counts(ascending=True)

df.shape

df = df.dropna()
df.shape

#Basic information of the data
df.date = pd.to_datetime(df.date)
df.date.dtype

start_time = min(df.date).strftime('%d %B %Y')
end_time = max(df.date).strftime('d %B %Y')
no_of_horses = df.horse_id.nunique()
no_of_races = df.race_id.nunique()

print(f'The dataset was collected from {start_time} to {end_time}, which contains information about {no_of_horses} horses and {no_of_races} races. ')

#drop the unnecessary columns
df = df.drop(columns=['horse_id','date'])
df.head()
df.columns

from time import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn.metrics import precision_score, classification_report, confusion_matrix

#Extract the last race data for model testing
last_raceid = max(df.race_id)
last_raceid

last_race = df[df.race_id == last_raceid]
last_race

new_data = df[:75696]
new_data = new_data.drop(columns='race_id')
new_data.tail()

new_data.shape

#Distribution of labels
plt.figure(figsize=(6,4))
sns.countplot(data=new_data, x='won')
plt.title('Number of labels by Class')
X = new_data.drop(columns='won')
y = new_data['won']

y.value_counts()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

k_range = range(1,10)
scores = {}
scores_list = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores[k] = precision_score(y_test, y_pred)
    scores_list.append(precision_score(y_test, y_pred))

import operator
max(scores.items(), key=operator.itemgetter(1))

plt.plot(k_range, scores_list)
plt.xlabel('Value of K for KNN')
plt.ylabel('precision_score of the positive class')
plt.title('Original Data')

start = time()

knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

end = time()
running_time = end - start
print('time cost: %.5f sec' %running_time)

print(classification_report(y_test, y_pred))

labels = ['lose', 'win']
cm = confusion_matrix(y_test, y_pred)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('confusion_matrix')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')

lgb.plot_importance(clf_sm, max_num_features=10)
