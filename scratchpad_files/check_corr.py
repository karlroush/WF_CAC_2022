# -*- coding: utf-8 -*-
"""
//
"""

import pandas as pd
import numpy as np

import tensorflow as tf

#%% Load data
#first pull raw training data
df= pd.read_excel('CAC+2022_Training+Data+Set+New.xlsx')
print(df.head()) #checking to make sure data is loaded 

#%% Clean the data
print(list(df))

print(df.loc[df['payment_category'] == 'Card'])


list_old = list(map(str, df.payment_category.tolist()))
list_new = list(map(str, df.db_cr_cd.tolist()))

df['new_claim'] = df.Category.replace(to_replace=['Claim ID: ' + e for e in list_old], value=['Claim ID: ' + e for e in list_new], regex=True)

#%% Data exploration as seperate cell (to prevent constant loading)
# dfcorr = df.corr()
# print(dfcorr)
df['new'] = np.where((df['Column1'] <= df['Column2']) & (
    df['Column1'] <= df['Column3']), df['Column1'], np.nan)

#%%
# card_df= df.loc[df['payment_category']== 'Card']
# print(card_df['db_cr_cd'])

# df2 = card_df.where(df['payment_category']=='Card', df['db_cr_cd'])
# print(df2)