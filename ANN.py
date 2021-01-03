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
#print(X_train.shape, X_test.shape)
#print(y_train.shape,y_test.shape)

##ANN
##create ANN model
import keras
from keras.models import Sequential
from keras.layers import Lambda, Dense, ReLU, LeakyReLU, Dropout

model = Sequential()
#input layer:
model.add(Dense(128, kernel_initializer= 'normal', input_dim= X_train.shape[1], activation= 'relu'))
##Hidden Layer:
model.add(Dense(256, kernel_initializer= 'normal', activation= 'relu'))
model.add(Dense(256, kernel_initializer= 'normal', activation= 'relu'))
model.add(Dense(256, kernel_initializer= 'normal', activation= 'relu'))

#Output layer
model.add(Dense(1, kernel_initializer= 'normal', activation= 'linear'))

#Compile the model
model.compile(loss= 'mean_absolute_error', optimizer= 'adam', metrics = ['mean_absolute_error'])

print(model.summary())

##Now train the model using fit method
result = model.fit(X_train, y_train, validation_split = 0.3, batch_size = 10, epochs = 100)

#model evaluation
prediction = model.predict(X_test)

sns.distplot(y_test.values.reshape(-1,1) - prediction)
plt.show()

plt.scatter(y_test, prediction)
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error
print("MAE: ", mean_absolute_error(y_test, prediction))
print("MSE: ", mean_squared_error(y_test, prediction))
print("RMSE: ", np.sqrt(mean_squared_error(y_test, prediction)))

