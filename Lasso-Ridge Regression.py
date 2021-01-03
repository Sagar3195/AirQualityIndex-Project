import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##Loading dataset
df = pd.read_csv("Data/Real_Data/Real_combine.csv")
print(df.head())
print(df.shape)

##Check null values in dataset
print(df.isnull().sum())

#sns.heatmap(df.isnull(),yticklabels= False, cbar= False, cmap = 'viridis')
#plt.show()
##Now we drop null values from dataset
df = df.dropna()
print(df.isnull().sum())

print(df.shape)
#sns.heatmap(df.isnull(), yticklabels= False, cbar = False, cmap= 'viridis')
#plt.show()

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
#print(X.head())
#print(y.head())

from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)

print(model.feature_importances_)

##Let's plot the graph of feature importance for better visualization
feat_importance = pd.Series(model.feature_importances_, index = X.columns)
#feat_importance.nlargest(5).plot(kind = 'barh')
#plt.show()

##Target variable visualization
#sns.displot(y)
#plt.show()

##Now we split dataset into training and testing data
from sklearn.model_selection import train_test_split
X_train,  X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state= 0)
print(X_train.shape, X_test.shape)
print(y_train.shape,y_test.shape)


##Comparision Betweeen Linear, Ridge ad Lasso Regression
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score

regressor = LinearRegression()
scores = cross_val_score(regressor, X,y, scoring = 'neg_mean_squared_error', cv = 5)
mean_scores = np.mean(scores)
print(mean_scores)

#Ridge Regression
ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1,5,10, 20, 30, 35, 40]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring= 'neg_mean_squared_error', cv = 5)
ridge_regressor.fit(X,y)

print(ridge_regressor.best_params_)

print(ridge_regressor.best_score_)

##Lasso Regression
lasso = Lasso()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1,5,10,20,30,35,40]}
lasso_regressor = GridSearchCV(lasso, parameters, scoring = 'neg_mean_squared_error', cv = 5)

lasso_regressor.fit(X,y)

print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

##Now we predict the model on test data
prediction = lasso_regressor.predict(X_test)

#sns.distplot(y_test - prediction)
#plt.show()

#plt.scatter(y_test, prediction)
#plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error
print("MAE: ", mean_absolute_error(y_test, prediction))
print("MSE: ", mean_squared_error(y_test, prediction))
print("RMSE: ", np.sqrt(mean_squared_error(y_test, prediction)))


import pickle
##Open a file, where you want to store the data.
#file = open("lasso_regression_model.pkl", 'wb')

#Dump the model
#pickle.dump(lasso_regressor, file)

