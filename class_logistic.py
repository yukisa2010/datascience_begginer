import numpy as np
import pandas as pd
from pandas import Series,DataFrame

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

iris = load_iris()


X = iris.data
Y = iris.target

iris_data = DataFrame(X,columns=['Sepal Length','Sepal Width','Petal Length','Petal Width'])
iris_target = DataFrame(Y, columns=['Species'])

def flower(num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'VerisColor'
    else:
        return 'Verginica'


iris_target['Species'] = iris_target['Species'].apply(flower)


iris = pd.concat([iris_data,iris_target], axis=1)


# sns.pairplot(iris, hue='Species', height=2,diag_kind='hist')
# sns.pairplot(iris, hue='Species', height=2,diag_kind='auto')

# plt.figure(figsize=(12,4))
# sns.countplot('Petal Length', data=iris, hue='Species')


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


logreg = LogisticRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.4,random_state=3)

from sklearn import metrics

logreg.fit(X_train,Y_train)
Y_pred = logreg.predict(X_test)


# print(metrics.accuracy_score(Y_test,Y_pred))

# k近傍法

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc = metrics.accuracy_score(Y_test,Y_pred)
print(acc)

# plt.show()

# kを1-60まで変化させてみる

k_range = range(1,90)
accuracy = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,Y_train)
    Y_pred = knn.predict(X_test)
    accuracy.append(metrics.accuracy_score(Y_test,Y_pred))


plt.plot(k_range,accuracy)
plt.show()