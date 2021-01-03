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

from sklearn.neighbors import KNeighborsRegressor
knn_1 = KNeighborsRegressor(n_neighbors= 1)
knn_1.fit(X_train, y_train)

print("Coefficient of determination R^2 <-- on train set: {}".format(knn_1.score(X_train, y_train)))

print("Coefficient of determination R^2 <-- on train set: {}".format(knn_1.score(X_test, y_test)))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn_1, X,y, cv = 5)
print(scores.mean())
##Prediction on test data
knn_1_predict = knn_1.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error
print("MAE of KNN_1: ", mean_absolute_error(y_test, knn_1_predict))
print("MSE of KNN_1: ", mean_squared_error(y_test, knn_1_predict))
print("RMSE of KNN_1: ", np.sqrt(mean_squared_error(y_test, knn_1_predict)))
##Model evaluation
#y_predict = knn.predict(X_test)

#sns.distplot(y_test - y_predict)
#plt.show()

#plt.scatter(y_test, y_predict)
#plt.show()

##Hyperparameter Tunning
# accuracy_rate = []
# for i in range(1, 30):
#     knn = KNeighborsRegressor(n_neighbors= i)
#     scores = cross_val_score(knn, X,y,cv = 10, scoring = 'neg_mean_squared_error')
#     accuracy_rate.append(scores.mean())
#
# plt.figure(figsize=(10, 8))
# plt.plot(range(1,30), accuracy_rate, color = 'blue', linestyle = 'dashed', marker = 'o',
#          markerfacecolor = 'red', markersize = 10)
# plt.title("Accuracy Rate vs Knn Value")
# plt.xlabel("K")
# plt.ylabel("Accuracy Rate")
# plt.show()
#

knn_3 = KNeighborsRegressor(n_neighbors= 3)
knn_3.fit(X_train, y_train)

print("Coefficient of determination R^2 <-- on train set: {}".format(knn_3.score(X_train, y_train)))

print("Coefficient of determination R^2 <-- on train set: {}".format(knn_3.score(X_test, y_test)))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn_3, X,y, cv = 5)
print(scores.mean())

knn_3_predict = knn_3.predict(X_test)

print("MAE of KNN_3: ", mean_absolute_error(y_test, knn_3_predict))
print("MSE of KNN_3: ", mean_squared_error(y_test, knn_3_predict))
print("RMSE of KNN_3: ", np.sqrt(mean_squared_error(y_test, knn_3_predict)))








