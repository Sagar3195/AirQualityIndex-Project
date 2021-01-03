"""Apply ML Algorithms :
   1. Linear Regression
   2. Lasso and Ridge Regression
   3. Decision Tree Regressor
   4. KNN Regressor
   5. RandomForest Regressor
   6. Xgboost Regressor
   7. Hyperparameter Tunning
   8. ANN"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##After data collection, next step is Feature Engineering , we will check null values , visulization of data
## Drop unwanted null values from dataframe

##Lodaing dataset
df = pd.read_csv("Data/Real_Data/Real_combine.csv")
print(df.head(10))
print("Shape of the dataset: ", df.shape)
##Check null values in dataset
print(df.isnull().sum())
##We can see that there is null values in dataset
#sns.heatmap(df.isnull(), yticklabels= False, cbar = False, cmap= 'viridis')
#plt.show()
##now we drop null values
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

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
##Train the model
regressor.fit(X_train, y_train)

print("Coefficient : ",regressor.coef_)

print("Intercept: ", regressor.intercept_)

print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_train, y_train)))

print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_test, y_test)))

from sklearn.model_selection import cross_val_score

score = cross_val_score(regressor, X,y, cv = 5)
print(score)
print("Score: ",score.mean())

##Model Evaluation

coeff_df = pd.DataFrame(regressor.coef_, index= X.columns, columns = ['Coefficient'])
print(coeff_df.head(10))
###Interpreting the Coefficients
"""1. Holding all features fixed, a 1 unit increase in T with an decrease of 10.12 in AQI PM 2.5
   2. Holding all features fixed, a 1 unit increase in TM with an increase of 3.92 in AQI PM 2.5
   3. Holding all features fixed, a 1 unit increase in VV with an decrease of 47.20 in AQI PM 2.5"""


##Prediction on test data
prediction = regressor.predict(X_test)
sns.displot(y_test - prediction)
#plt.show()

#plt.scatter(y_test, prediction)
#plt.show()

"""Regression Evaluation Metrics :-
Here are three common evaluation metrics for regression problems:

Mean Absolute Error (MAE) is the mean of the absolute value of the errors

Mean Squared Error (MSE) is the mean of the squared errors.
Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors.

Comparing these metrics:

MAE is the easiest to understand, because it's the average error.
MSE is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units.

All of these are loss functions, because we want to minimize them."""
from sklearn.metrics import mean_squared_error, mean_absolute_error
print("MAE: ", mean_absolute_error(y_test, prediction))
print("MSE: ",mean_squared_error(y_test, prediction))
print("RMSE: ", np.sqrt(mean_squared_error(y_test, prediction)))

import pickle
##Open a file , where we want to store the data
file = open("regression.pkl", 'wb')
##dump the information to that file
pickle.dump(regressor, file)
