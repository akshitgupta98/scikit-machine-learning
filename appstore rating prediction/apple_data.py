import numpy as np
import pandas as pd

datax = pd.read_csv("appleStore_description.csv")
data  =pd.read_csv("AppleStore.csv")

y= datax.iloc[:,2:3].values
x = data.iloc[:,[3,5,6,7,8,9,13,14,15,16]].values



from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(x[:,:])
x[:, :] = imputer.transform(x[:, :])
imputer = imputer.fit(y[:,:])
y[:, :] = imputer.transform(y[:, :])

from sklearn.cross_validation import train_test_split
xtrain , xtest , ytrain , ytest = train_test_split(x, y, test_size = 0.2,random_state = 0)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(xtrain,ytrain)
ypred = rf.predict(xtest)
rf.score(xtrain,ytrain)#0.99

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(xtrain,ytrain)
ypred1 = dt.predict(xtest)
dt.score(xtrain,ytrain) #1.0

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(xtrain,ytrain)
ypred2 = lr.predict(xtest)
lr.score(xtrain,ytrain)#1.0


ytrain = ytrain.ravel()
from sklearn.svm import SVR
sv = SVR(kernel='rbf')
sv.fit(xtrain,ytrain)
ypred3 = sv.predict(xtest)
sv.score(xtrain,ytrain)#-0.08