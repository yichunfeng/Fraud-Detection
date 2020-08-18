#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 20:25:32 2020

@author: yvonne
"""
import gc
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import toad
import category_encoders
import copy
from scipy.stats import ks_2samp

# resetting parameter
sns.set()
warnings.filterwarnings('ignore')

# Reducing the memory usage in pandas
def reduce_mem_usage(df):
     #iterate through all the columns of a dataframe and modify the data type
        #to reduce memory usage.        
    
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object :
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# Loading data
folder_path = '/Users/yvonne/Desktop/Fraud-Detection/input/'
print('Loading data...')

train_identity = pd.read_csv(f'{folder_path}train_identity.csv', index_col='TransactionID')
print('\tSuccessfully loaded train_identity!')

train_transaction = pd.read_csv(f'{folder_path}train_transaction.csv', index_col='TransactionID')
print('\tSuccessfully loaded train_transaction!')

test_identity = pd.read_csv(f'{folder_path}test_identity.csv', index_col='TransactionID')
print('\tSuccessfully loaded test_identity!')

test_transaction = pd.read_csv(f'{folder_path}test_transaction.csv', index_col='TransactionID')
print('\tSuccessfully loaded test_transaction!')

sub = pd.read_csv(f'{folder_path}sample_submission.csv')
print('\tSuccessfully loaded sample_submission!')

print('Data was successfully loaded!\n')

#train=pd.DataFrame()
#test=pd.DataFrame()


# merge
train_identity['has_id']=1
train=train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test_identity['had_id']=1
test=test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)


    
cats=['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4',
 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16',
 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24',
 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32',
 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']



cats_test=['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5',
 'card6', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2',
 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id-12', 'id-13', 'id-14', 'id-15',
 'id-16', 'id-17', 'id-18', 'id-19', 'id-20', 'id-21', 'id-22', 'id-23', 'id-24',
 'id-25', 'id-26', 'id-27', 'id-28', 'id-29', 'id-30', 'id-31', 'id-32',
 'id-33', 'id-34', 'id-35', 'id-36', 'id-37', 'id-38', 'DeviceType', 'DeviceInfo']




print('Reducing memory...')
train[cats]=train[cats].astype(str)
test[cats_test]=test[cats_test].astype(str)
train=reduce_mem_usage(train)
test=reduce_mem_usage(test)
print('Finished reducing memory: ')
print("\ttrain shape:{}, test shape:{}".format(train.shape, test.shape))


print('Sampling...\n')
X = copy.deepcopy(train)
X_test = copy.deepcopy(test)
# target
y = X["isFraud"]
del X["isFraud"]

print("X shape:{}, X_test shape:{}".format(X.shape, X_test.shape))
print("y shape:{}".format(y.shape))






# Caculating PSI
NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS)

columns = X.drop(cats,axis=1).columns
splits = folds.split(X, y)

PSI = pd.DataFrame()
PSI['feature'] = columns
  
for fold_n, (train_index, valid_index) in enumerate(splits):
    X_train, X_valid =X.iloc[train_index],X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    

    
    PSI[f'fold_{fold_n + 1}'] = toad.metrics.PSI(X_train.drop(cats,axis=1),X_valid.drop(cats,axis=1)).values
    
    
    gc.collect()
    
print('PSI: ',PSI)


# Negative sampling
print('Negative sampling...\n')
X['y']=y
fraud=X[X.y==1]



# Checking the missing value of card1 ~ card6 to determine whether they can be mapped to unique users
card1 = fraud.card1.value_counts()
print('number of missing value in card1: ',fraud.card1.isnull().sum())
card1 = card1[card1>0]
print('number of unique card1: ',card1.shape[0])

card2 = fraud.card2.value_counts()
print('number of missing value in card2: ',fraud.card2.isnull().sum())
card2 = card2[card2>0]
print('number of unique card2: ',card2.shape[0])

card3 = fraud.card3.value_counts()
print('number of missing value in card3: ',fraud.card3.isnull().sum())
card3 = card3[card3>0]
print('number of unique card3: ',card3.shape[0])

card4 = fraud.card4.value_counts()
print('number of missing value in card4: ',fraud.card4.isnull().sum())
card4 = card4[card4>0]
print('number of unique card4: ',card4.shape[0])

card5 = fraud.card5.value_counts()
print('number of missing value in card5: ',fraud.card5.isnull().sum())
card5 = card5[card5>0]
print('number of unique card5: ',card5.shape[0])

card6 = fraud.card6.value_counts()
print('number of missing value in card6: ',fraud.card6.isnull().sum())
card6 = card6[card6>0]
print('number of unique card6: ',card6.shape[0],'\n')



# Checking the missing value of addr1, addr2
fraud.addr1 = fraud.addr1.replace('nan',np.nan)
fraud.addr2 = fraud.addr2.replace('nan',np.nan)
print('number of missing value in addr1: ', fraud.addr1.isnull().sum())
print('number of missing value in addr2: ', fraud.addr2.isnull().sum(),'\n')


# Checking the missing value of emails
print('number of missing value in P_emaildomain: ',fraud['P_emaildomain'].replace('None',np.nan).isnull().sum())
print('number of missing value in R_emaildomain: ', fraud['R_emaildomain'].replace('None',np.nan).isnull().sum())


'''

params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 1,#0.3797454081646243,
          'bagging_fraction': 1,#0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47,
          'num_threads':6
         }
NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS)

columns = X.columns
splits = folds.split(X, y)
y_preds = np.zeros(X_test.shape[0])
y_oof = np.zeros(X.shape[0])
score = 0

feature_importances = pd.DataFrame()
feature_importances['feature'] = columns
  
for fold_n, (train_index, valid_index) in enumerate(splits):
    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=100)
    
    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()
    
    y_pred_valid = clf.predict(X_valid)
    y_oof[valid_index] = y_pred_valid
    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")
    
    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS
    #y_preds += clf.predict(X_test) / NFOLDS
    
    del X_train, X_valid, y_train, y_valid
    gc.collect()
    
score=score*NFOLDS
y_preds=y_preds*NFOLDS
print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")


# feature importance
feature_importances['average'] = feature_importances[[f'fold_{fold_n+1}' for fold_n in range(5)]].mean(axis = 1)
plt.figure(figsize = (8,10))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average',y='feature')
plt.title('50 TOP feature importance over 5 folds average'.format(folds.n_splits))

'''


# Binning of TransactionAmt

# Function for caculating PSI of TransactionAmt
def PSI_cal(NFOLDS,X,y,cats):
    NFOLDS = NFOLDS
    folds = StratifiedKFold(n_splits=NFOLDS)

    columns = X.drop(cats,axis=1).columns
    splits = folds.split(X, y)

    PSI = pd.DataFrame()
    PSI['feature'] = columns

    for fold_n, (train_index, valid_index) in enumerate(splits):
        X_train, X_valid =X.iloc[train_index],X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]



        #PSI[f'fold_{fold_n + 1}'] = toad.metrics.PSI(X_train.drop(cats,axis=1),X_valid.drop(cats,axis=1)).values

        PSI[f'fold_{fold_n + 1}'] = toad.metrics.PSI(X_train.TransactionAmt,X_valid.TransactionAmt)
        del X_train, X_valid, y_train, y_valid
        gc.collect()
    return PSI.drop('feature',axis=1).iloc[0].mean()


# IV & PSI before binning

print('Class: TransactionAmt')
print('Original Information Value: ',toad.quality(X[['TransactionAmt','y']],'y').iv.values[0])
print('Original Population Stability Index: ',PSI_cal(5,X,y,'TransactionAmt'))



# Binning - cart tree
iv=[]
PSIs=[]
TransactionAmt=X.TransactionAmt
for bins in [5,10,15,20,25,30,35,40,45,50]:
    bins=toad.DTMerge(TransactionAmt,y,n_bins=bins).tolist()
    bins.insert(0,-np.inf)
    bins.append(np.inf)
    X.TransactionAmt=np.digitize(TransactionAmt,bins)
    iv.append(toad.quality(X[['TransactionAmt','y']],'y').iv.values[0])
    PSIs.append(PSI_cal(5,X,y,cats))

print('IV after cart tree: ',iv)
print('PSI after cart tree: ',PSIs)



# Binning - chimerge
iv=[]
PSIs=[]
for bins in [5,10,15,20,25,30,35,40,45,50]:
    bins=toad.ChiMerge(TransactionAmt,y,n_bins=bins).tolist() # DTmerge，ChiMerge，KMeansMerge，QuantileMerge
    bins.insert(0,-np.inf)
    bins.append(np.inf)
    X.TransactionAmt=np.digitize(TransactionAmt,bins)
    iv.append(toad.quality(X[['TransactionAmt','y']],'y').iv.values[0])
    PSIs.append(PSI_cal(5,X,y,cats))

print('IV after chimerge: ',iv)
print('PSI after chimerge: ',PSIs)



# Binning - kmeans merge
iv=[]
PSIs=[]
for bins in [5,10,15,20,25,30,35,40,45,50]:
    bins=toad.KMeansMerge(TransactionAmt,y,n_bins=bins).tolist() # DTmerge，ChiMerge，KMeansMerge，QuantileMerge
    bins.insert(0,-np.inf)
    bins.append(np.inf)
    X.TransactionAmt=np.digitize(TransactionAmt,bins)
    iv.append(toad.quality(X[['TransactionAmt','y']],'y').iv.values[0])
    PSIs.append(PSI_cal(5,X,y, cats))

print('IV after kmeans merge: ',iv)
print('PSI after kmeans merge: ',PSIs)


'''

The effect is best when using cart tree with bins = 10, but the training result is not as expected


# The magical treatment for TransactionAmt provided in the discussion: 
  the decimal point of the transaction amount determines countries!
  Therefore, I deprecate binning.

bins=toad.ChiMerge(TransactionAmt,y,n_bins=10).tolist() # DTmerge，ChiMerge，KMeansMerge，QuantileMerge
bins.insert(0,-np.inf)
bins.append(np.inf)
X.TransactionAmt=np.digitize(TransactionAmt,bins)

params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 1,#0.3797454081646243,
          'bagging_fraction': 1,#0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47,
          'num_threads':6
         }
NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS)

columns = X.columns
splits = folds.split(X, y)
y_preds = np.zeros(X_test.shape[0])
y_oof = np.zeros(X.shape[0])
score = 0

feature_importances = pd.DataFrame()
feature_importances['feature'] = columns
  
for fold_n, (train_index, valid_index) in enumerate(splits):
    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=100)
    
    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()
    
    y_pred_valid = clf.predict(X_valid)
    y_oof[valid_index] = y_pred_valid
    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")
    
    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS
    #y_preds += clf.predict(X_test) / NFOLDS
    
    del X_train, X_valid, y_train, y_valid
    gc.collect()
    

print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")

'''

