'''
Building a binary classifier on one label
'''

import pandas as pd
import json
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#%% Load data
#first pull raw training data
df= pd.read_excel('CAC+2022_Training+Data+Set+New.xlsx', index_col=0)
print(df.head()) #checking to make sure data is loaded 

#%% Breaking code into cells to prevent re-loading data

#checking to make sure headers and labels are as expected
# print(list(df))
# print(len(df['Category'].unique()), df['Category'].unique()) #10 types

#split into binary data
cleaned_training= df.loc[df['Category'].isin(['Finance', 'Services to Transport'])]
print(cleaned_training['Category'].unique())


trans_type= []
trans_data= []
for index, row in cleaned_training.iterrows():
    trans_type.append(row['Category'])
    
    data_items=" ".join([str(row['merchant_cat_code']), str(row['amt']),
                         str(row['payment_category']), str(row['coalesced_brand'])])
    trans_data.append(data_items)
    
#%% Build tokenizer
training_size= int(len(trans_data)*0.8)
training_sentences= trans_data[0:training_size]
testing_sentances= trans_data[training_size:]
training_labels= trans_type[0:training_size]
testing_labels= trans_type[training_size:]

tokenizer= Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(training_sentences)

word_index= tokenizer.word_index

training_sequences= tokenizer.texts_to_sequences(training_sentences)
training_padded= pad_sequences(training_sequences, padding= 'post')

testing_sequences= tokenizer.texts_to_sequences(testing_sentances)
testing_padded= pad_sequences(testing_sequences, padding= 'post')

#%% Build Model
