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
#plt.figure(figsize = (20, 20))
#sns.heatmap(df[top_corr_features].corr(), annot= True, cmap= 'RdYlGn')
#plt.show()


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
#conda install -c ananconda py-xgboost
import xgboost as xgb
xgb_regressor = xgb.XGBRegressor()

xgb_regressor.fit(X_train, y_train)

print("Coefficient of determination R^2 <-- on train set: {}".format(xgb_regressor.score(X_train, y_train)))

print("Coefficient of determination R^2 <-- on train set: {}".format(xgb_regressor.score(X_test, y_test)))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(xgb_regressor, X,y, cv = 5)
print(scores)

##Model evaluation
y_predict = xgb_regressor.predict(X_test)

#sns.distplot(y_test - y_predict)
#plt.show()

#plt.scatter(y_test, y_predict)
#plt.show()

##Hyperparameter Tunning
from sklearn.model_selection import RandomizedSearchCV
##Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
##Various learning rate parameters
learning_rate = ['0.05', '0.1', '0.2', '0.3', '0.5', '0.6']
##Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

##Subsample parameter value
subsample = [0.7, 0.6, 0.8]
##Minimum child weight parameters
min_child_weight = [3,4,5,6,7]

##create a random grid
param_grid = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth,
              'subsample': subsample, 'min_child_weight': min_child_weight}

##we use random grid to search best hyperparameters.
xgb_random = RandomizedSearchCV(estimator= xgb_regressor, param_distributions= param_grid, scoring = 'neg_mean_squared_error',
                               n_iter= 100, cv = 5, verbose= 2, random_state= 42, n_jobs= 1)

xgb_random.fit(X_train, y_train)
print("Best parameters: ",xgb_random.best_params_)

print("Best scores: ", xgb_random.best_score_)

prediction = xgb_random.predict(X_test)

#sns.distplot(y_test - prediction)
#plt.show()

#plt.scatter(y_test, prediction)
#plt.show()

##Metrics evaluations
from sklearn.metrics import mean_absolute_error, mean_squared_error
print("MSE: ", mean_squared_error(y_test, prediction))
print("MAE: ", mean_absolute_error(y_test, prediction))
print("RMSE: ", np.sqrt(mean_squared_error(y_test,prediction)))


import pickle
# Open a file where we want to store the data
#file = open('Xgboost_regressor.pkl', 'wb')

##dump information to that file
#pickle.dump(xgb_random, file)




