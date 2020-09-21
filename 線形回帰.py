import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

from sklearn.datasets import load_boston
from sklearn.preprocessing import add_dummy_feature

boston = load_boston()

# print(boston.DESCR)

# plt.hist(boston.target,bins=50)
# plt.xlabel('Price($1,000)')
# plt.ylabel('Number of houses')


# plt.scatter(boston.data[:,5],boston.target)

# plt.ylabel('Price($1,000)')
# plt.xlabel('Number of rooms')

boston_df = DataFrame(boston.data)
boston_df.columns = boston.feature_names


# print(boston_df.head())
# plt.scatter(boston.data[:,6],boston.target)


boston_df['Price'] = boston.target

# sns.lmplot('RM','Price',data=boston_df)



# plt.show()

# Part2

X = boston_df.RM

# print(boston_df.RM)
X = np.vstack(boston_df.RM)

# print(X[:4])
# print(X)
# print(X.shape)

Y = np.array(boston_df.Price)
X = np.array([[value,1] for value in X],dtype=np.float64)
# return
# X = np.array([[value[0],1] for value in X])
# print(X[:4])
# print(Y[:4])
# X = add_dummy_feature(X)

# print(X[:5])
# print(np.linalg.lstsq(X,Y))

# print(X.dtype,Y.dtype)
# print(X.shape,Y.shape)

# a , b= np.linalg.lstsq(X,Y)[0]
# print(a,b)

# y=a*x + b

# plt.plot(boston_df.RM,boston_df.Price,'o')

# x = boston_df.RM
# plt.plot(x,a*x+b,'r')


# Part3

result = np.linalg.lstsq(X,Y,rcond=None)

error_total = result[1]
rmse = np.sqrt(error_total/len(X))
print('平均二乗誤差の平方根={:0.2f}'.format(rmse[0]))


# 重回帰分析

import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_multi = boston_df.drop('Price',1)

# print(X_multi)

Y_target = boston_df.Price

# lreg.fit(X_multi,Y_target)

# print(lreg.intercept_)
# print(lreg.coef_)
# plt.show()


coeff_df = DataFrame(boston_df.columns)

# coeff_df.columns = ['Features']
# coeff_df['Coefficient Estimate'] = pd.Series(lreg.coef_)
# print(coeff_df) #係数

X_train, X_test, Y_train, Y_test = train_test_split(X_multi,Y_target)

lreg = LinearRegression()
lreg.fit(X_train,Y_train)

pred_train = lreg.predict(X_train)
pred_test = lreg.predict(X_test)

# print(np.mean((Y_train - pred_train)**2),'train')
# print(np.mean((Y_test - pred_test)**2),'test')
print(pred_train[0])

train = plt.scatter(pred_train,(pred_train-Y_train),c='b',alpha=0.5)
test = plt.scatter(pred_test,(pred_test-Y_test),c='r',alpha=0.5)
plt.hlines(y=0,xmin=1.0,xmax=50)

plt.legend((train,test),('Training','Test'),loc='lower left')
plt.title('Residual Plots')



# plt.show()

