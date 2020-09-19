import pandas as pd
from pandas import Series, DataFrame
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

import requests
from io import StringIO

url = 'http://elections.huffingtonpost.com/pollster/2012-general-election-romney-vs-obama.csv'

source = requests.get(url).text

poll_data = StringIO(source)
poll_df = pd.read_csv(poll_data)

# print(poll_df.head())

poll_df[['Pollster','Partisan','Affiliation']].sort_values('Pollster').drop_duplicates()

# sns.countplot('Affiliation',data=poll_df,hue='Population',order=['Rep','Dem','None'])
# print(ans)

avg = pd.DataFrame(poll_df.mean())

avg.drop('Number of Observations',axis=0,inplace=True)

std = pd.DataFrame(poll_df.std())
std.drop('Number of Observations',axis=0,inplace=True)

# avg.plot(yerr=std,kind='bar',legend=False)

poll_avg = pd.concat([avg,std],axis=1)
poll_avg.columns = ['Average','STD']

# print(poll_df)

# poll_df.plot(x='End Date', y=['Obama','Romney','Undecided'],marker='o',linestyle='')

from datetime import datetime

poll_df['Difference'] = (poll_df.Obama - poll_df.Romney)/100

poll_df = poll_df.groupby(['Start Date'],as_index=False,).mean()
print(poll_df.head())
# plt.show()