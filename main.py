import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

path = kagglehub.dataset_download("camnugent/california-housing-prices")
print("Path to dataset files:", path)

csv_path = os.path.join(path, "housing.csv")

data = pd.read_csv(csv_path)
data.dropna(inplace=True)

from sklearn.model_selection import train_test_split

X = data.drop(['median_house_value'], axis=1)
Y = data['median_house_value']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#turn to gaussian curve
# TODO Make this a function
train_data = X_train.join(Y_train)
train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population'] = np.log(train_data['population'] + 1)
train_data['households'] = np.log(train_data['households'] + 1)

# flags for ocean proximity
train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)

#plt.figure(figsize=(15,8))
#sns.scatterplot(x="latitude", y="longitude", data=train_data, hue="median_house_value", palette="coolwarm")
#plt.show()

#feature engineering
train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms'] #we have rooms and bedrooms amount => how many of those rooms are bedrooms as a new feature
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households'] # how many rooms per household?

#plt.figure(figsize=(15,8))
#sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")
#plt.show()

#simple linear regression model training
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train, Y_train = train_data.drop(['median_house_value'], axis=1), train_data['median_house_value']
X_train_s = scaler.fit_transform(X_train)

reg = LinearRegression()
reg.fit(X_train_s, Y_train)

#evaluate model
test_data = X_test.join(Y_test)

test_data['total_rooms'] = np.log(test_data['total_rooms'] + 1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1)
test_data['population'] = np.log(test_data['population'] + 1)
test_data['households'] = np.log(test_data['households'] + 1)

test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)

test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms'] #we have rooms and bedrooms amount => how many of those rooms are bedrooms as a new feature
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households'] # how many rooms per household?

if 'ISLAND' not in test_data.columns:
    col_index = train_data.columns.get_loc("ISLAND")
    test_data.insert(col_index, "ISLAND", False)

print(train_data)
print(test_data)

X_test, Y_test = test_data.drop(['median_house_value'], axis=1), test_data['median_house_value']
X_test_s = scaler.transform(X_test)

print(reg.score(X_test_s, Y_test))

#random forest regressor
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(X_train_s, Y_train)
print(forest.score(X_test_s, Y_test))

#hyper parameter training
from sklearn.model_selection import GridSearchCV

forest = RandomForestRegressor()

param_grid = {
    'n_estimators': [100, 200, 300],
    'min_samples_split': [2, 4],
    'max_depth': [None, 4, 8]
}

grid_search = GridSearchCV(forest, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
grid_search.fit(X_train_s, Y_train)

best_forest = grid_search.best_estimator_
print(best_forest)
print(best_forest.score(X_test_s, Y_test))