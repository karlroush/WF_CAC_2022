# -*- coding: utf-8 -*-
"""
Main model file
"""

#%% ==================== IMPORT LIBRARIES ==================== %%#  
# Custom written files
from data_cleaning import cleanData

# Library imports
import pandas as pd
import numpy as np
import time
from os.path import exists

# ML related library imports
from sklearn.model_selection import train_test_split
from autoviml.Auto_ViML import Auto_ViML
from autoviml.Auto_NLP import plot_confusion_matrix, plot_classification_matrix

from sklearn.utils import class_weight
# import tensorflow_hub as hub
start_time = time.time() #track how long program takes to run

#%% ==================== CLEAN AND LOAD DATA ==================== %%#  

# Clean initial raw data if the cleaned data file does not exist
if not(exists('./cleaned_data/clean_training_data.xlsx')):
    df= pd.read_excel('./provided_data/CAC+2022_Training+Data+Set+New.xlsx')
    c_df= cleanData(df) #see report for rationale
    c_df.to_excel('./cleaned_data/clean_training_data.xlsx', index= False)
    del c_df #to save memory space
    
if not(exists('./cleaned_data/clean_test_data.xlsx')):
    df= pd.read_excel('./provided_data/CAC+2022_Test+Data+Set+New.xlsx')
    c_df= cleanData(df) #see report for rationale
    c_df.to_excel('./cleaned_data/clean_test_data.xlsx', index= False)
    del c_df #to save memory space
    
#load the cleaned data
df = pd.read_excel('./cleaned_data/clean_training_data.xlsx')
test_df= pd.read_excel('./cleaned_data/clean_test_data.xlsx')

d= {'Communication Services': 0, 'Education': 1, 'Entertainment': 2,
    'Finance': 3, 'Health and Community Services': 4, 'Property and Business Services': 5,
    'Retail Trade': 6, 'Services to Transport': 7, 'Trade, Professional and Personal Services': 8,
    'Travel': 9
    }
df['target'] = df['Category'].map(d) #create numeric target based on category
df= df.fillna(-1)

#apply same logic to test data
test_df=test_df.fillna(-1)


#%% ==================== COMPUTE CLASS WEIGHTS (IMBALANCED) ==================== %%#  

#weighting the different classes of data since categories are unevenly distributed
class_weights= list(class_weight.compute_class_weight('balanced',
                    classes= np.unique(df['Category']), 
                    y= df['Category']))
class_weights.sort()
weights= {}
for index, weight in enumerate(class_weights):
    weights[index]= weight
print(weights)


#%% ==================== SPLIT DATA AND TRAIN MODEL ==================== %%#  
#split data into train and test
train, test= train_test_split(df.loc[:,df.columns != 'Category'], test_size=0.2) 
target= 'target' #identify target (the category converted to number)

m, feats, trainm, testm= Auto_ViML(train, target, test, 
                                   sample_submission='', scoring_parameter='',
                                   KMeans_Featurizer=False, hyper_param= 'RS',
                                   feature_reduction=True,
                                   Boosting_Flag='CatBoost', Binning_Flag=False,
                                   Add_Poly=0, Stacking_Flag= False, Imbalanced_Flag= True,
                                   verbose= 2)


#%% ==================== MAKE PREDICTIONS ON TEST DATA ==================== %%#  
m.save_model('model', format= 'cbm') #save model
# plot_confusion_matrix(test[target].values, m.predict(testm[feats]))

#Pre-process test data
test_df= test_df[feats] #pull out features the model uses
test_df= test_df.fillna(-1) #replace and NaN
test_df['merchant_cat_code'] = test_df['merchant_cat_code'].astype(int)
test_df['amt'] = test_df['amt'].astype(int)

#make predictions
preds= m.predict(test_df)


#%% ==================== SAVE PREDICTIONS IN DESIRED FORMAT ==================== %%#  
# add predictions to test data 
final_data= pd.read_excel('./provided_data/CAC+2022_Test+Data+Set+New.xlsx')
final_data['target']= preds

# Map the numbers back to category labels
d= dict([(value, key) for key, value in d.items()])
print(d)
final_data['Category'] = final_data['target'].map(d) 
# final_data = final_data.drop(columns=['target'])

# save predictions on test data to file
# final_data.to_excel('categorized_test_data.xlsx', index= False)


#%% ==================== COMPARE TO UNMODIFIED CATBOOST MODEL ==================== %%#  
from catboost import Pool, CatBoostClassifier

# change the compare boolean to True if running comparison
compare= False 

if compare:
    # train_data= train.loc[:, train.columns != 'target']
    # eval_data= test.loc[:, test.columns != 'target']
    train_data= train[feats]
    eval_data= test[feats]
    
    train_data= train_data.astype(str)
    eval_data= eval_data.astype(str)
    train_label = train['target']
    eval_label = test['target']
    
    cat_features= feats #identify features to train on
    train_dataset = Pool(data=train_data,
                          label=train_label,
                          cat_features=cat_features)
    
    eval_dataset = Pool(data=eval_data,
                        label=eval_label,
                        cat_features=cat_features)
    
    # Initialize CatBoostClassifier
    model = CatBoostClassifier(iterations=100,
                                learning_rate=1,
                                depth=4,
                                loss_function='MultiClass')
    # Fit model
    model.fit(train_dataset)
    
    # Get predicted classes
    preds_class = model.predict(eval_dataset)
    preds2= model.predict(test_df)
    
    # Compare predictions with novel approach
    print((sum(preds==preds2)/len(preds))[0])
    
    # Get predicted probabilities for each class
    preds_proba = model.predict_proba(eval_dataset)
    # Get predicted RawFormulaVal
    preds_raw = model.predict(eval_dataset,
                              prediction_type='RawFormulaVal')


print("\n--- %s seconds ---" % (time.time() - start_time))