# -*- coding: utf-8 -*-
"""
Initial data explorations and cleaning of raw data
"""
import pandas as pd
import numpy as np
import time


def loadData(filename):
    df= pd.read_excel(filename)
    print(df.head()) #checking to make sure data is loaded 
    return df

def cleanData(dataframe):
    #delete irrelevant columns
    c_df = dataframe.drop(columns=['sor', 'cdf_seq_no','trans_desc',
                                   'payment_reporting_category', 
                                   'default_brand','qrated_brand'])
    
    #Now we need to modify the payment_category, default_location
    #use the values in db_cr_cd to fix "Card" in payment_category
    c_df['db_cr_cd']= c_df['db_cr_cd'].fillna('X') #get rid of NaN
    c_df['combined']= c_df['db_cr_cd'] + c_df['payment_category'] #join two
    combos= c_df['combined'].unique() #get unique list of joins
    # print(combos)
    combos_clean = [x if x != 'DCard' else 'DDebit Card' for x in combos]
    combos_clean = list(map(lambda x: x.replace('CCard', 'CCredit Card'), combos))
    # print(combos_clean)
    combos_clean = [sub[1 : ] for sub in combos_clean]
    d = dict(zip(combos,combos_clean)) #map the combined "DDebit Card" etc to just "Debit Card" etc
    d['DCard']='Debit Card'
    # print(d)
    
    c_df['combined'] = c_df['combined'].map(d) #replace with correct descriptions
    c_df['payment_category']=  c_df['combined'] #overwrite initial payment method col
    c_df = c_df.drop(columns=['combined']) #delete now useless cols
    c_df = c_df.drop(columns=['db_cr_cd'])
    
    #default location modification
    c_df['default_location'] = c_df['default_location'].str.replace('\d+', '', regex= True) #get rid of numbers
    c_df['default_location'] = c_df['default_location'].str.replace('-+', '', regex= True) #get rid of dashes
    c_df['default_location'] = c_df['default_location'].str.strip() #remove leading and trailing spaces
    return c_df

if __name__ == '__main__':
    start_time = time.time()
    #%% ========== LOAD DATA ========== %%# 
    #first pull raw data
    df= loadData('./provided_data/CAC+2022_Training+Data+Set+New.xlsx')
    
    #%% ========== Checking which fields to remove ========== %%# 
    print(list(df))
    num_entries= df[df.columns[0]].count()

    print('\nSOR info: ', df['sor'].value_counts()) #check values for sor
    print(df['sor'].value_counts()[0]/num_entries) #percentage of HH items

    print('\nMerchant cat code info: ', df['merchant_cat_code'].isnull().sum())
    print(df['merchant_cat_code'].isnull().sum()/num_entries) #percentage of merchants w/ no ID

    df['new'] = np.where(df['qrated_brand'] != df['coalesced_brand'], df['coalesced_brand'], np.nan)
    print('\nBrand field differences: ', (num_entries- df['new'].isna().sum() )/num_entries) #number of items differing between brand fields
    del df['new']
    
    print('\n', df['payment_reporting_category'].value_counts()) #checking to see if there is one value for payment reporting

    #%% ========== Cleaning data and removing cols ========== %%# 
    c_df= cleanData(df) #see report for rationale
    c_df.to_excel('./cleaned_data/clean_training_data.xlsx', index= False)  
    
    print("\n--- %s seconds ---" % (time.time() - start_time))