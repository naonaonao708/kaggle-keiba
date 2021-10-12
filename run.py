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
