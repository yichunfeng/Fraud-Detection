#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 00:09:07 2020

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


print('Cross-validation...')
params = {'num_leaves': 491,
          'feature_fraction': 1,#0.3797454081646243,
          'bagging_fraction': 1,#0.4181193142567742,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.1,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'random_state': 47,
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

    clf = lgb.train(params, dtrain, 1000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=100)
    
    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()
    
    y_pred_valid = clf.predict(X_valid)
    y_oof[valid_index] = y_pred_valid
    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")
    
    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS
    y_preds += clf.predict(X_test) / NFOLDS
    
    del X_train, X_valid, y_train, y_valid
    gc.collect()

    
    
print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")

# feature importance
feature_importances['average'] = feature_importances[[f'fold_{fold_n+1}' for fold_n in range(5)]].mean(axis = 1)
plt.figure(figsize = (8,10))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average',y='feature')
plt.title('50 TOP feature importance over 5 folds average'.format(folds.n_splits))
plt.savefig('feature importance of Cross-validation')

print('Cross-validation Finished \n')
print('\n')
print('\n')


# dealing with time
# Deleting TransactionDT, and replacing it with the derived features
print('Dealing with time...\n')
X['time'] = pd.to_datetime(X['TransactionDT']+1546272000, unit='s')#2019.01.01 0:0:0偏移
X['year'] = X["time"].dt.year
X['month'] = X["time"].dt.month
X['day'] = X["time"].dt.day
X['hour']  = X["time"].dt.hour
X['minute']  = X["time"].dt.minute
X['weekday']  = X["time"].dt.dayofweek
X.pop('time')
X.drop(['year','month'],axis=1,inplace=True)

X_test['time'] = pd.to_datetime(X_test['TransactionDT']+1546272000, unit='s')#2019.01.01 0:0:0偏移
X_test['year'] = X_test["time"].dt.year
X_test['month'] = X_test["time"].dt.month
X_test['day'] = X_test["time"].dt.day
X_test['hour']  = X_test["time"].dt.hour
X_test['minute']  = X_test["time"].dt.minute
X_test['weekday']  = X_test["time"].dt.dayofweek
X_test.pop('time')
X_test.drop(['year','month'],axis=1,inplace=True)

del X['TransactionDT']
del X_test['TransactionDT']



print('After dealing with TransactionDT...')
params = {'num_leaves': 491,
          'feature_fraction': 1,#0.3797454081646243,
          'bagging_fraction': 1,#0.4181193142567742,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.1,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'random_state': 47,
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

    clf = lgb.train(params, dtrain, 1000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=100)
    
    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()
    
    y_pred_valid = clf.predict(X_valid)
    y_oof[valid_index] = y_pred_valid
    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")
    
    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS
    y_preds += clf.predict(X_test) / NFOLDS
    
    del X_train, X_valid, y_train, y_valid
    gc.collect()

    
    
print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")

# feature importance
feature_importances['average'] = feature_importances[[f'fold_{fold_n+1}' for fold_n in range(5)]].mean(axis = 1)
plt.figure(figsize = (8,10))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average',y='feature')
plt.title('50 TOP feature importance over 5 folds average'.format(folds.n_splits))
plt.savefig('feature importance After dealing with TransactionDT')

print('Dealing with TransactionDT Finished \n')
print('\n')
print('\n')


# Dealing with TransactionAmt
print('Dealing with TransactionAmt..')
X['TransactionAmt_decimal'] = ((X['TransactionAmt'] - X['TransactionAmt'].astype(int)) * 1000).astype(int)
X['TransactionAmt'] = X['TransactionAmt_decimal']
del X['TransactionAmt_decimal']


params = {'num_leaves': 491,
          'feature_fraction': 1,#0.3797454081646243,
          'bagging_fraction': 1,#0.4181193142567742,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.1,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'random_state': 47,
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

    clf = lgb.train(params, dtrain, 1000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=100)
    
    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()
    
    y_pred_valid = clf.predict(X_valid)
    y_oof[valid_index] = y_pred_valid
    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")
    
    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS
    y_preds += clf.predict(X_test) / NFOLDS
    
    del X_train, X_valid, y_train, y_valid
    gc.collect()

    
    
print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")

# feature importance
feature_importances['average'] = feature_importances[[f'fold_{fold_n+1}' for fold_n in range(5)]].mean(axis = 1)
plt.figure(figsize = (8,10))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average',y='feature')
plt.title('50 TOP feature importance over 5 folds average'.format(folds.n_splits))
plt.savefig('feature importance After dealing with TransactionAmt')

print('Dealing with TransactionAmt Finished \n')
print('\n')
print('\n')



print('Aggregation of card, addr and email...')
X['identity']=X.card1.astype(str)+'_'+X.card2.astype(str)+'_'+X.card3.astype(str)+'_'+ \
X.card4.astype(str)+'_'+X.card5.astype(str)+'_'+X.card6.astype(str)

X.identity=X.identity.astype(str)+'_'+X.addr1.astype(str)+'_'+X.addr2.astype(str)+'_'+X.P_emaildomain.astype(str)+'_'+X.R_emaildomain.astype(str)
X.identity=X.identity.astype('category')


params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
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

    clf = lgb.train(params, dtrain, 1000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=100)
    
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




# feature importance
feature_importances['average'] = feature_importances[[f'fold_{fold_n+1}' for fold_n in range(5)]].mean(axis = 1)
plt.figure(figsize = (8,10))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average',y='feature')
plt.title('50 TOP feature importance over 5 folds average'.format(folds.n_splits))
plt.savefig('feature importance After Aggregation')

print('Aggregation Finished \n')
print('\n')
print('\n')


