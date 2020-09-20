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
print(X[:4])
print(Y[:4])
# X = add_dummy_feature(X)

# print(X[:5])
# print(np.linalg.lstsq(X,Y))

print(X.dtype,Y.dtype)
print(X.shape,Y.shape)

a , b= np.linalg.lstsq(X,Y)[0]
print(a,b)

# y=a*x + b

# plt.plot(boston_df.RM,boston_df.Price,'o')

# x = boston_df.RM
# plt.plot(x,a*x+b,'r')


# Part3








# plt.show()

