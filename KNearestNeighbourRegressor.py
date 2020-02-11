# In this program we will use KNN to predict air quality index

# Part 1: Preprocessing

# Importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the data
df=pd.read_csv('Real_Combine.csv')
# Checking for null values
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Since there are very few null values, we can drop them
df = df.dropna()
# Defining dependent and independent features
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Seeing the feature importance using ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)

# Plotting the importance using ExtraTreesRegressor
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

# Splitting the data using train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Part 2: Making K nearest neighbour regressor

# Importing the library
from sklearn.neighbors import KNeighborsRegressor

# Making the model with k value = 1
regressor=KNeighborsRegressor(n_neighbors=1)
# Fitting the data into the model
regressor.fit(X_train,y_train)

# Printing coeficient of determination of the training set
print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_train, y_train)))
# Printing coeficient of determination of the testing set
print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_test, y_test)))
# The value for training set is 1, for testing set is 0.54, the model is underfitting

# Printing cross validation score for regressor model with 5 fold cross validation
from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,X,y,cv=5)
score.mean()
# The mean cross validation score is 0.39

# Part 3: Prediction and evaluation

# Making the prediction
prediction=regressor.predict(X_test)

# Plotting the difference in the test value and the predicted value
sns.distplot(y_test-prediction)
# Very low kurtosis
# Plotting the predicted values against the test values
plt.scatter(y_test,prediction)
# More or less linear with some outliers

# Part 4: Hyper parameter tuning

# We start a loop to take n value from 1 to 40 and give the mean cross validation score with 10 folds
accuracy_rate = []
for i in range(1,40):
    knn = KNeighborsRegressor(n_neighbors=i)
    score=cross_val_score(knn,X,y,cv=10,scoring="neg_mean_squared_error")
    accuracy_rate.append(score.mean())

# Plotting the cross validation score against the n value to arruve at the best n value
plt.figure(figsize=(10,6))
plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plt.title('Accuracy Rate vs. K Value')
plt.xlabel('K')
# Accuracy rate is highest at n=1

# Part 6: Prediction and evalution using the tuned model

# Initiating the model with one neighbour
knn = KNeighborsRegressor(n_neighbors=1)
# Fitting the data onto the model
knn.fit(X_train,y_train)
# Making the predictions
predictions = knn.predict(X_test)

# Plotting the difference of the test value and the predicted values
sns.distplot(y_test-predictions)
# Extremely high kurtosis, this is good

# Plotting the predicted values against the test values
plt.scatter(y_test,predictions)
# Very linear with several outliers

# Printing the MAE, MSE and RMSE
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

# Part 6: Dumping the model in a pickle file for later deployment

# Importing pickkle
import pickle

# opening the file to dump the model
file = open('random_forest_regression_model', 'wb')
# dumping the tuned model into the file
pickle.dump(knn, file)




