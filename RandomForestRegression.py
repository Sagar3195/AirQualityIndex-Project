import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

##Loading dataset
df = pd.read_csv("Data/Real_Data/Real_combine.csv")
print(df.head())
print(df.shape)

##Check null values in dataset
print(df.isnull().sum())
sns.heatmap(df.isnull(), yticklabels= False, cbar = False, cmap= 'viridis')
plt.show()

##Now we drop null values from dataset
df = df.dropna()
print(df.isnull().sum())

print(df.shape)


##Let's visualize the correlations between features
#sns.pairplot(df)
#plt.show()

#print(df.corr())

##Correlation Matrix with Heatmap
"""1. Correlation states how the features are related to each others or the target variable.
2. Correlation can be positive (increase in one value of feature increases value of target variable) or
   negative (increase in value of one feature decreases the value of the target variables).
3. Heatmap makes it easy to identify which features are most related to the target variables , 
   we will plot heatmap of correlated features using the seaborn library."""
##Let's visualize the corrlation of each feature using heatmap
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (20, 20))
sns.heatmap(df[top_corr_features].corr(), annot= True, cmap= 'RdYlGn')
plt.show()


### Feature Importance
"""1. You can get the feature importance of each feature of your dataset by using the 
feature importance property of the model.
2. Feature importance gives you a score for each feature of your data, the higher the score more important 
or relevant is the feature towards your output variable.
3. Feature importance is an inbuilt class that comes with Tree Based Regressor, we will be using 
Extra Tree Regressor for extracting the top 10 feature for the dataset."""
##Splitting dataset into Independent features and dependent features
X = df.iloc[:, :-1]
y = df.iloc[:,-1]
print(X.head())
print(y.head())

from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)

print(model.feature_importances_)

##Now we split dataset into training and testing data
from sklearn.model_selection import train_test_split
X_train,  X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state= 0)
print(X_train.shape, X_test.shape)
print(y_train.shape,y_test.shape)

##RandomForest Regression
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()
forest.fit(X_train, y_train)

print("Coefficient of determination R^2 <-- on train set: {}".format(forest.score(X_train, y_train)))


print("Coefficient of determination R^2 <-- on train set: {}".format(forest.score(X_test, y_test)))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest, X,y, cv = 5)
print(scores.mean())

#Prediction model on test data
#prediction = forest.predict(X_test)

#sns.distplot(y_test - prediction)
#plt.show()

#plt.scatter(y_test, prediction)
#plt.show()

#Hyperparameter Tunning
from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
#print(n_estimators)
##Random Search CV
#Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
##Number of features to consider at every split
max_features = ['auto', 'sqrt']
##Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(start = 5, stop = 30, num = 6)]
#Minimum number of samples required to split a node
min_samples_split = [2,5,10, 15, 100]
##Minimum number of samples required at each leaf node
min_samples_leaf = [1,2,5,10]

##create random grid
random_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
#print(random_grid)

rf_random = RandomizedSearchCV(estimator= forest, param_distributions= random_grid, scoring = 'neg_mean_squared_error',
                               n_iter = 100, cv = 5, verbose= 2, n_jobs= 1, random_state= 42)

rf_random.fit(X_train, y_train)

print("Best parameters: ",rf_random.best_params_)
print("Best scores: ", rf_random.best_score_)

##Now prediction model on test data
predictions = rf_random.predict(X_test)

#sns.displot(y_test - predictions)
#plt.show()

#plt.scatter(y_test, predictions)
#plt.show()

##Regression Evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
print("MAE: ", mean_absolute_error(y_test, predictions))
print("MSE: ", mean_squared_error(y_test, predictions))
print("RMSE: ", np.sqrt(mean_squared_error(y_test, predictions)))

import pickle
##Open a file where we want to store data
#file = open("random_forest_regression.pkl", 'wb')

##dump information on that file
#pickle.dump(rf_random, file)
