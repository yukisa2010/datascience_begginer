import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
Y = iris.target

from sklearn.svm import SVC

model = SVC()

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

model.fit(X_train,Y_train)
predicted = model.predict(X_test)

from sklearn import metrics

ac = metrics.accuracy_score(Y_test,predicted)
print(ac)
