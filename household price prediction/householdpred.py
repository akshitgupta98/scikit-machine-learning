import pandas as pd

train = pd.read_csv("test.csv")
test = pd.read_csv("train.csv")

train1 = train.iloc[: , [0,1,3,4,17,18,19,20,26,34,36,37,38,43,44,45,46,47,48,49,50,51,52,54,56,59,61,62,66,67,68,69,70,71,75,76,77]].values
test1 = test.iloc[0:1459 , [0,1,3,4,17,18,19,20,26,34,36,37,38,43,44,45,46,47,48,49,50,51,52,54,56,59,61,62,66,67,68,69,70,71,75,76,77]].values
ytrain = test.iloc[0:1459 , 80:81].values




from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(train1[:,:])
train1[:, :] = imputer.transform(train1[:, :])
imputer = imputer.fit(test1[:,:])
test1[:, :] = imputer.transform(test1[:, :])
imputer = imputer.fit(ytrain[:,:])
ytrain[:, :] = imputer.transform(ytrain[:, :])




'''from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss = ss.fit(train1[:,:])
train1[:, :] = ss.transform(train1[:, :])
ss = ss.fit(test1[:,:])
test1[:, :] = ss.transform(test1[:, :])
ss = ss.fit(ytrain[:,:])
ytrain[:, :] = ss.transform(ytrain[:, :])'''

ytrain = ytrain.ravel()


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(train1 , ytrain)
ypred = dt.predict(test1) #0.64



from sklearn.ensemble import RandomForestRegressor
rt = RandomForestRegressor()
rt.fit(train1 , ytrain)
ypred1 = dt.predict(test1)#0.64




from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train1, ytrain)
ypred2 = lr.predict(test1)#0.55


from sklearn.svm import SVR
sv = SVR(kernel = 'rbf')
sv.fit(train1 , ytrain)
ypred3 = sv.predict(test1)