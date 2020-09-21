import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import math

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn import metrics

import statsmodels.api as sm

def logistic(t):
    return 1.0/(1 + math.exp((-1.0)*t))

t = np.linspace(-6,6,500)

y = np.array([logistic(ele) for ele in t])

# plt.plot(t,y)
# plt.title('Logistic Function')

df = sm.datasets.fair.load_pandas().data


def affair_check(x):
    if x != 0:
        return 1
    else:
        return 0

df['Had_Affair'] = df['affairs'].apply(affair_check)
# print(df)

mean = df.groupby('Had_Affair').mean()
# print(mean)


# sns.countplot('age',data=df.sort_values('age'),hue='Had_Affair',palette='coolwarm')
# sns.countplot('yrs_married',data=df.sort_values('yrs_married'),hue='Had_Affair',palette='coolwarm')
# sns.countplot('children',data=df.sort_values('children'),hue='Had_Affair',palette='coolwarm')
# sns.countplot('religious',data=df.sort_values('religious'),hue='Had_Affair',palette='coolwarm')


acc_dummies = pd.get_dummies(df.occupation)
hus_occ_dummies = pd.get_dummies(df.occupation_husb)


acc_dummies.columns = ['acc1','acc2','acc3','acc4','acc5','acc6']
hus_occ_dummies.columns = ['hocc1','hocc2','hocc3','hocc4','hocc5','hocc6']


X = df.drop(['occupation','occupation_husb','Had_Affair'],axis=1)

dummies = pd.concat([acc_dummies,hus_occ_dummies],axis=1)

X = pd.concat([X,dummies],axis=1)
Y = df.Had_Affair

X = X.drop('acc1',axis=1)
X = X.drop('hocc1',axis=1)
X = X.drop('affairs',axis=1)


# Y = Y.values
Y = np.ravel(Y)






# print(Y.tail())
# plt.show()