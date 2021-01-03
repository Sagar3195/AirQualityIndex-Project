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

sns.heatmap(df.isnull(),yticklabels= False, cbar= False, cmap = 'viridis')
plt.show()
##Now we drop null values from dataset
df = df.dropna()
print(df.isnull().sum())

print(df.shape)

sns.heatmap(df.isnull(), yticklabels= False, cbar = False, cmap= 'viridis')
plt.show()

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
#print(X.head())
#print(y.head())

from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)

print(model.feature_importances_)

##Let's plot the graph of feature importance for better visualization
feat_importance = pd.Series(model.feature_importances_, index = X.columns)
feat_importance.nlargest(5).plot(kind = 'barh')
plt.show()

##Target variable visualization
sns.displot(y)
plt.show()


##Now we split dataset into training and testing data
from sklearn.model_selection import train_test_split
X_train,  X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state= 0)
print(X_train.shape, X_test.shape)
print(y_train.shape,y_test.shape)

##Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

dtree = DecisionTreeRegressor(criterion= 'mse')
dtree.fit(X_train, y_train)

print("Coefficient of determination R^2 <-- on train set: {}".format(dtree.score(X_train, y_train)))

print("Coefficient of determination R^2 <-- on test set: {}".format(dtree.score(X_test, y_test)))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(dtree, X,y, cv = 5)
print(scores.mean())

#Predict model on test data
prediction = dtree.predict(X_test)

sns.distplot(y_test - prediction)
sns.scatterplot(y_test, prediction)
plt.show()

##Hyperparameter Tunning Decision TreeRegressor
##Hyperparameter optimization using GridSearchCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
params = {'splitter': ['best', 'random'], 'max_depth': [3,4,5,6,8,10, 12,15],
          "min_samples_leaf": [1,2,3,4,5],
          "min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4],
          "max_features": ["auto", "log2","sqrt", None],
          "max_leaf_nodes": [None, 10, 20, 30, 40, 50, 60, 70]}

random_search = GridSearchCV(dtree, param_grid= params, scoring= 'neg_mean_squared_error',n_jobs= -1, cv = 10, verbose = 3)
##How much time take to train this model for that we created timer function

from datetime import datetime
import time
def timer(start_time = None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print("\n Time taken: %i hours %i minutes and %s seconds."% (thour,tmin, round(tsec, 2)))

##here we start
start_time = timer(None)  # timing starts from this point for "start_time" variable
random_search.fit(X,y)
timer(start_time) # timing ends here for "start_time" variable

print(random_search.best_score_)
print(random_search.best_params_)

#Prediction on test data
predictions = random_search.predict(X_test)

sns.distplot(y_test - predictions)
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error
print("MAE: ", mean_absolute_error(y_test, predictions))
print("MSE: ", mean_squared_error(y_test, predictions))
print("RMSE: ", np.sqrt(mean_squared_error(y_test, predictions)))


import pickle
##Open a file where we want to store the data
#file = open("decision_tree_model.pkl", 'wb')

##dump information to that file
#pickle.dump(random_search, file)
























