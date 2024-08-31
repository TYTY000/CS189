from sklearn.datasets import fetch_california_housing
# as_frame=True loads the data in a dataframe format, with other metadata besides it
california_housing = fetch_california_housing(as_frame=True)
# Select only the dataframe part and assign it to the df variable
df = california_housing.frame
import pandas as pd
print(df.head())
y = df['MedHouseVal']
X = df.drop(['MedHouseVal'], axis = 1)
print(X.describe().T)
from sklearn.model_selection import train_test_split

SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Fit only on X_train
scaler.fit(X_train)

# Scale both X_train and X_test
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=5)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'mae: {mae}')
print(f'mse: {mse}')
print(f'rmse: {rmse}')

regressor.score(X_test, y_test)

y.describe()

# error = []

# Calculating MAE error for K values between 1 and 39
# for i in range(1, 40):
#     knn = KNeighborsRegressor(n_neighbors=i)
#     knn.fit(X_train, y_train)
#     pred_i = knn.predict(X_test)
#     mae = mean_absolute_error(y_test, pred_i)
#     error.append(mae)

import matplotlib.pyplot as plt 

# plt.figure(figsize=(12, 6))
# plt.plot(range(1, 40), error, color='red', 
#          linestyle='dashed', marker='o',
#          markerfacecolor='blue', markersize=10)
#          
# plt.title('K Value MAE')
# plt.xlabel('K Value')
# plt.ylabel('Mean Absolute Error')
# plt.show()

import numpy as np
# k = np.array(error).argmin()
# print("min error = "+ str(error[k]))               # 0.43631325936692505
# print("index = "+str(k)) # 11

knn_reg12 = KNeighborsRegressor(n_neighbors=12)
knn_reg12.fit(X_train, y_train)
y_pred12 = knn_reg12.predict(X_test)
r2 = knn_reg12.score(X_test, y_test) 

mae12 = mean_absolute_error(y_test, y_pred12)
mse12 = mean_squared_error(y_test, y_pred12)
rmse12 = mean_squared_error(y_test, y_pred12, squared=False)
print(f'\nr2: {r2}, \nmae: {mae12} \nmse: {mse12} \nrmse: {rmse12}\n')

df["MedHouseValCat"] = pd.qcut(df["MedHouseVal"], 4, retbins=False, labels=[1, 2, 3, 4])
y = df['MedHouseValCat']
X = df.drop(['MedHouseVal', 'MedHouseValCat'], axis = 1)

from sklearn.model_selection import train_test_split

SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

acc =  classifier.score(X_test, y_test)
print(acc) # 0.6191860465116279

from sklearn.metrics import classification_report, confusion_matrix
#importing Seaborn's to use the heatmap 
import seaborn as sns

# Adding classes names for better interpretation
classes_names = ['class 1','class 2','class 3', 'class 4']
cm = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                  columns=classes_names, index = classes_names)
                  
# Seaborn's heatmap to better visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d');
plt.show()

print(classification_report(y_test, y_pred))

from sklearn.metrics import f1_score

f1s = []

# # Calculating f1 score for K values between 1 and 40
# for i in range(1, 40):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train, y_train)
#     pred_i = knn.predict(X_test)
#     # using average='weighted' to calculate a weighted average for the 4 classes 
#     f1s.append(f1_score(y_test, pred_i, average='weighted'))

# plt.figure(figsize=(12, 6))
# plt.plot(range(1, 40), f1s, color='red', linestyle='dashed', marker='o',
#          markerfacecolor='blue', markersize=10)
# plt.title('F1 Score K Value')
# plt.xlabel('K Value')
# plt.ylabel('F1 Score')
# plt.show()

classifier15 = KNeighborsClassifier(n_neighbors=15)
classifier15.fit(X_train, y_train)
y_pred15 = classifier15.predict(X_test)
print(classification_report(y_test, y_pred15))

from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors = 5)
nbrs.fit(X_train)
# Distances and indexes of the 5 neighbors 
distances, indexes = nbrs.kneighbors(X_train)

print(indexes[:3], indexes[:3].shape)

# dist_means = distances.mean(axis=1)
# plt.plot(dist_means)
# plt.title('Mean of the 5 neighbors distances for each data point')
# plt.xlabel('Count')
# plt.ylabel('Mean Distances')

dist_means = distances.mean(axis=1)
plt.plot(dist_means)
plt.title('Mean of the 5 neighbors distances for each data point with cut-off line')
plt.xlabel('Count')
plt.ylabel('Mean Distances')
plt.axhline(y = 3, color = 'r', linestyle = '--')

import numpy as np

# Visually determine cutoff values > 3
outlier_index = np.where(dist_means > 3)
print(outlier_index)

outlier_values = df.iloc[outlier_index]
print(outlier_values)




