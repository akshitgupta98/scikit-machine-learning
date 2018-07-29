import pandas as pd
import numpy as np

titanic = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

titanic.head()

titanic.info()

test.info()

d = {'male': 1, 'female': 0}
titanic['Sex'] = titanic['Sex'].map(d)
test['Sex'] = test['Sex'].map(d)
e = {"S":0 , "C":1 , "Q":2}
titanic['Embarked'] = titanic["Embarked"].map(e)
test['Embarked'] = test['Embarked'].map(e)


titanic = titanic.drop([ 'Name' , 'Ticket'] , axis = 1)
test = test.drop(['Name' , 'Ticket'] , axis = 1)


titanic.drop("Cabin",axis=1,inplace=True)
test.drop("Cabin",axis=1,inplace=True)


xtrain = titanic.drop(["Survived" ] , axis=1)
ytrain = titanic["Survived"]
xtest = test

xtrain = np.array(xtrain)
xtest = np.array(xtest)
ytrain = np.array(ytrain)



from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(xtrain[:,:])
xtrain[:, :] = imputer.transform(xtrain[:, :])

imputer = imputer.fit(xtest[:,:])
xtest[:,:] = imputer.transform(xtest[:, :])

from sklearn.ensemble import RandomForestClassifier


randomforest = RandomForestClassifier(n_estimators = 1000)
randomforest.fit(xtrain,ytrain)
ypred = randomforest.predict(xtest)
randomforest.score(xtrain,ytrain) #1.0 #0.74


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(xtrain, ytrain)

Y_pred = logreg.predict(xtest)

logreg.score(xtrain, ytrain)#0.80



from sklearn.svm import SVC

svc = SVC()

svc.fit(xtrain, ytrain)

Y__pred = svc.predict(xtest)

svc.score(xtrain, ytrain)#0.99 #0.62





from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(xtrain, ytrain)

Y___pred = knn.predict(xtest)

knn.score(xtrain, ytrain) #0.81







