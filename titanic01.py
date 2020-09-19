import pandas as pd
from pandas import DataFrame, Series

titanic_df = pd.read_csv('train.csv')

# titanic_df = titanic_df.pivot('Pclass','Survived')
# print(titanic_df.info())

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sns.countplot('Sex',data=titanic_df,hue='Pclass')
# sns.countplot('Pclass',data=titanic_df,hue='Sex')

def male_female_child(passenger):
    age, sex = passenger
    if age < 16:
        return 'child'
    else:
        return sex

titanic_df['person'] = titanic_df[['Age', 'Sex']].apply(male_female_child, axis=1)

# sns.countplot('Pclass',data=titanic_df,hue='person')

# titanic_df['Age'].hist(bins=70)
# print(titanic_df['Age'].mean())

# print(titanic_df['person'].value_counts())

# fig = sns.FacetGrid(titanic_df, hue='Pclass', aspect=4)
# fig.map(sns.kdeplot, 'Age', shade=True)
# oldest = titanic_df['Age'].max()
# fig.set(xlim=(0,oldest))
# fig.add_legend()

deck = titanic_df['Cabin'].dropna()
# print(type(deck))

levels = []

for level in deck:
    levels.append(level[0])

# print(levels)

cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
# print(cabin_df)

# sns.countplot('Cabin',data=cabin_df,palette='winter_d',order=sorted(set(levels)))
# sns.countplot('Embarked',data=titanic_df, hue='Pclass')

# cabin_df = cabin_df[cabin_df.Cabin != 'T']
# sns.countplot('Cabin',data=cabin_df,palette='summer',order=sorted(set(cabin_df.Cabin)))

from collections import Counter
Counter(titanic_df.Embarked)

# print(titanic_df.Embarked.value_counts())

titanic_df['Alone'] = titanic_df.Parch + titanic_df.SibSp
titanic_df['Alone'].loc[titanic_df['Alone']>0] = 'With family'
titanic_df['Alone'].loc[titanic_df['Alone']==0] = 'Alone'

# sns.countplot('Alone',data=titanic_df,palette='Blues')

titanic_df['Survivor'] = titanic_df.Survived.map({0:'no', 1:'yes'})

# sns.countplot('Survivor',data=titanic_df,palette='Set1')

# sns.factorplot('Pclass','Survived',data=titanic_df,order=[1,2,3])
# sns.factorplot('Pclass','Survived',hue='person',data=titanic_df,order=[1,2,3],aspect=2)
# print(titanic_df.head())

# sns.lmplot('Age','Survived',data=titanic_df)
# sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',hue_order=[1,2,3])

generations = [10,20,40,60,80,100]
sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',
            x_bins=generations)

plt.show()