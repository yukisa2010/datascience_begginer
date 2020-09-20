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
# print(poll_df.head())
# fig = poll_df.plot('Start Date', 'Difference',figsize=(12,4),marker='o',linestyle='-',color='purple')

#10/3, 10/11, 10/22

# poll_df[poll_df['Start Date'].apply(lambda x: x.startswith('2012-10'))]

# # fig = poll_df.plot('Start Date', 'Difference',figsize=(12,4),marker='o',linestyle='-',color='purple',xlim=[325,352])

# # plt.axvline(x=326,linewidth=4,color='gray')
# # plt.axvline(x=333,linewidth=4,color='gray')
# # plt.axvline(x=343,linewidth=4,color='gray')


# Part3
# 寄付

# https://www.dropbox.com/s/l29oppon2veaq4n/Election_Donor_Data.csv?dl=0

donor_df = pd.read_csv('Election_Donor_Data.csv')
# print(donor_df['contb_receipt_amt'].value_counts().shape)

don_mean = donor_df['contb_receipt_amt'].mean()
don_std = donor_df['contb_receipt_amt'].std()


# print('平均{:0.2f} 標準偏差{:0.2f}'.format(don_mean,don_std))

top_donor = donor_df['contb_receipt_amt'].copy()

top_donor.sort_values(ascending=True,inplace=True)

top_donor = top_donor[top_donor > 0]
# print(top_donor.value_counts().head(10))

com_don = top_donor[top_donor < 2500]
# com_don.hist(bins=100)


# print(top_donor)

# print(donor_df.info())
# print(donor_df.head())

# Part4


candidates = donor_df.cand_nm.unique()
# print(candidates)

party_map = {
'Cain, Herman' :'Republican',
'Bachmann, Michelle':'Republican',
'Cain, Herman':'Republican',
'Gingrich, Newt':'Republican',
'Huntsman, Jon':'Republican',
'Johnson, Gary Earl':'Republican',
'McCotter, Thaddeus G':'Republican',
'Obama, Barack':'Democrat',
'Paul, Ron':'Republican',
'Pawlenty, Timothy' :'Republican',
'Perry, Rick':'Republican',
"Roemer, Charles E. 'Buddy' III":'Republican', 
'Romney, Mitt' :'Republican',
 'Santorum, Rick' :'Republican'
}

# print(donor_df.head())

donor_df['Party'] = donor_df.cand_nm.map(party_map)
donor_df = donor_df[donor_df.contb_receipt_amt > 0]

# print(donor_df.groupby('cand_nm')['contb_receipt_amt'].count())

cand_amount = donor_df.groupby('cand_nm')['contb_receipt_amt'].sum()
# cand_amount.plot(kind='bar')

# donor_df.groupby('Party')['contb_receipt_amt'].sum().plot(kind='bar')

occupation_df = donor_df.pivot_table('contb_receipt_amt',index='contbr_occupation',columns='Party',aggfunc='sum')
# print(occupation_df.head())
occupation_df = occupation_df[occupation_df.sum(1) > 1000000]

# print(occupation_df.shape)

# occupation_df.plot(kind='bar')

# plt.xticks(rotation =45,fontsize =5)

occupation_df.drop(['INFORMATION REQUESTED PER BEST EFFORTS'],axis=0,inplace=True)
occupation_df.loc['CEO'] = occupation_df.loc['CEO'] + occupation_df.loc['C.E.O.']

occupation_df.drop('C.E.O.',inplace=True)
occupation_df.plot(kind='barh',figsize=(10,12),cmap='seismic')

# print(occupation_df)
plt.show()