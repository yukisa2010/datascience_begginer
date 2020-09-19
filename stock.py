import pandas as pd
from pandas import Series, DataFrame
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_style('whitegrid')
from pandas_datareader.data import DataReader

from datetime import datetime

tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

end = datetime.now()
start = datetime(end.year -1, end.month, end.day)

for stock in tech_list:
    globals()[stock] = DataReader(stock,'yahoo',start,end)


# AAPL['Adj Close'].plot(legend=True,figsize=(10,4))

ma_day = [10,20,50]
# print(type(AAPL))
for ma in ma_day:
    column_name = 'MA {}'.format(ma)
    AAPL[column_name] = AAPL['Adj Close'].rolling(window=ma).mean()

# AAPL[['Adj Close','MA 10','MA 20','MA 50']].plot(subplots=False,figsize=(10,4))

AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()
# AAPL['Daily Return'].plot(figsize=(10,4),legend=True,linestyle='--',marker='o')
# print(AAPL.head())

# sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')
# AAPL['Daily Return'].hist(bins=100)

closing_df = DataReader(['AAPL','GOOG','MSFT','AMZN'], 'yahoo',start, end)['Adj Close']

tech_rets = closing_df.pct_change()

# sns.pairplot(tech_rets.dropna())

# sns.jointplot('GOOG','MSFT',tech_rets, kind='scatter',color='seagreen')
# returns_fig = sns.PairGrid(closing_df)
# returns_fig.map_upper(plt.scatter,color='purple')
# returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
# returns_fig.map_diag(plt.hist,bins=30)
# print(tech_rets.head())

# sns.heatmap(tech_rets.corr(),annot=True)

# Part4

rets = tech_rets.dropna()

# plt.scatter(rets.mean(),rets.std(),alpha=0.5,s=np.pi*20)

# plt.ylim([0.01,0.03])
# plt.xlim([-0.005,0.01])
# # print(rets.head())


# plt.xlabel('Expected Returns')
# plt.ylabel('Risk')

# for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
#     plt.annotate(label, xy=(x,y), xytext=(0,50),
#                 textcoords='offset points', ha='right',va='bottom',
#                 arrowprops=dict(arrowstyle='-',connectionstyle='arc3'))

# Part5

# sns.distplot(AAPL['Daily Return'].dropna(), bins=100, color='purple')

# a = rets['AAPL'].quantile(0.05)


days = 365
dt = 1/days
mu = rets.mean()['GOOG']
sigma = rets.std()['GOOG']

def stock_monte_carlo(start_price, days, mu, sigma):
    price = np.zeros(days)
    price[0] = start_price
    shock = np.zeros(days)
    drift = np.zeros(days)

    for x in range(1,days):
        shock[x] = np.random.normal(loc=mu*dt, scale=sigma * np.sqrt(dt))
        drift[x] = mu * dt
        price[x] = price[x-1] + (price[x-1]*(drift[x] + shock[x]))
    return price

# print(GOOG.head())


start_price = GOOG.iloc[0,5]





# for run in range(5):
#     plt.plot(stock_monte_carlo(start_price, days, mu, sigma))
# plt.xlabel('Days')
# plt.ylabel('Price')
# plt.title('Monte Carlo Analysis')


runs = 10000
simulations = np.zeros(runs)
np.set_printoptions(threshold=5)
for run in range(runs):
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]


plt.hist(simulations,bins=200)

q = np.percentile(simulations,1)
plt.hist(simulations,bins=200)
plt.figtext(0.6,0.8,s='Start price: {:0.2}'.format(start_price))
plt.figtext(0.6,0.7,'mean final price: {:0.2}'.format(simulations.mean()))
plt.figtext(0.6,0.6,'VaR(0.99): {:0.2f}'.format(start_price-q))
plt.figtext(0.15,0.6,'q(0.99): {:0.2f}'.format(q))

plt.axvline(x = q,linewidth=4,color='r')
plt.show() 