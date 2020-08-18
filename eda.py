#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 23:36:29 2020

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

# Resetting parameter
sns.set()
warnings.filterwarnings('ignore')


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

#sub = pd.read_csv(f'{folder_path}sample_submission.csv')
#print('\tSuccessfully loaded sample_submission!')

print('Data was successfully loaded!\n')


# Checking the time series of train data
train_feature_to_eda=None
train_feature_to_eda=train_transaction.TransactionDT
plt.figure(figsize=(8,5))
plt.title('TransactionDT in train_transaction')
plt.plot(train_feature_to_eda)

# Checking the time series of test data
test_feature_to_eda=None
test_feature_to_eda=test_transaction.TransactionDT
plt.figure(figsize=(8,5))
plt.title('TransactionDT in test_transaction')
plt.plot(test_feature_to_eda)


# KS test of TransactionDT in train and test

ks_result = ks_2samp(train_transaction['TransactionDT'].values,test_transaction['TransactionDT'].values)
print('KS Test of TransactionDT in train_trasaction and test_transaction: ',ks_result)


# Transforming time
train=pd.DataFrame()
train['time'] = pd.to_datetime(train_transaction['TransactionDT']+1546272000, unit='s')#2019.01.01 0:0:0偏移
train['year'] = train["time"].dt.year
train['month'] = train["time"].dt.month
train['day'] = train["time"].dt.day
train['hour']  = train["time"].dt.hour
train['minute']  = train["time"].dt.minute
train['weekday']  = train["time"].dt.dayofweek
train.pop('time')

test=pd.DataFrame()
test['time'] = pd.to_datetime(test_transaction['TransactionDT']+1546272000, unit='s')#2019.01.01 0:0:0偏移
test['year'] = test["time"].dt.year
test['month'] = test["time"].dt.month
test['day'] = test["time"].dt.day
test['hour']  = test["time"].dt.hour
test['minute']  = test["time"].dt.minute
test['weekday']  = test["time"].dt.dayofweek
test.pop('time')


# KS test of time
def corr(train, test,cols):
    # assuming first column is `class_name_id`


    for class_name in cols:
        # all correlations
        print('\n Class: %s' % class_name)

        ks_stat, p_value = ks_2samp(train[class_name].values,
                                    test[class_name].values)
        print(' Kolmogorov-Smirnov test:    KS-stat = %.6f    p-value = %.3e\n'
              % (ks_stat, p_value))

corr(train, test,cols=['year','month','day','hour','minute','weekday'])
corr(train, test,cols=['year','month','day','hour','minute','weekday'])



# The distribution deviations of month and year are relatively large
print('year in train: ',train.year.value_counts())
print('year in test: ',test.year.value_counts())
print('month in train: ',train.month.value_counts())
print('month in test: ', test.month.value_counts())
train.drop(['year','month'],axis=1,inplace=True)
test.drop(['year','month'],axis=1,inplace=True)

# Store the derived features of train.TransactionDT and test.TransactionDT
#train.to_csv(f'{folder_path}/derivation/train.TransactionDT_deriv.csv',index=False)
#test.to_csv(f'{folder_path}/derivation/test.TransactionDT_deriv.csv',index=False)



# Distribution of TransactionAmt
plt.figure(figsize=(8,5))
plt.title('TransactionAmt in train_transaction')
sns.kdeplot(train_transaction.TransactionAmt)

# Distribution of TransactionAmt - Logarithmic function
plt.figure(figsize=(8,5))
plt.title('TransactionAmt in train_transaction - Logarithm')
sns.kdeplot(np.log1p(train_transaction.TransactionAmt))

# Distribution of TransactionAmt - Box Plot
plt.figure(figsize=(8,5))
plt.title('TransactionAmt in train_transaction - Box Plot')
plt.boxplot(train_transaction.TransactionAmt)




# Analysing productCD
print('Number of Missing Values in ProductCD in train_transaction:',train_transaction.ProductCD.isnull().sum())

plt.figure(figsize=(8,5))
plt.title('Count of Observations by ProductCD')
train_transaction.groupby('ProductCD').size() \
    .sort_index() \
    .plot(kind='barh')




# Cumulative Distribution Function
def cumulate(data,col):
    print('Class: '+ col)
    print('Percentage of Missing Value: '+str(data[col].isnull().sum()/data.shape[0]))  
    num_category = data[col].value_counts().shape
    print('Number of Categories: '+str(num_category[0]),'\n')
    #print('Category Feature'+'\n'+str(data[col].value_counts()))
 
    tp=(data[col].value_counts()/data.shape[0]).values.flatten().tolist()
    cumulate=[]
    for i in range(1,len(tp)+1):
        cumulate.append(sum(tp[0:i]))
    return cumulate


# Analysing card1 ~ card6
plt.figure(figsize=(8,5))
plt.title('Cumulation of Card1')
c1 = cumulate(train_transaction,'card1')
plt.plot(c1)  

plt.figure(figsize=(8,5))
plt.title('Cumulation of Card2')
c2 = cumulate(train_transaction,'card2')
plt.plot(c2)  

plt.figure(figsize=(8,5))
plt.title('Cumulation of Card3')
c3 = cumulate(train_transaction,'card3')
plt.plot(c3)  

plt.figure(figsize=(8,5))
plt.title('Cumulation of Card4')
c4 = cumulate(train_transaction,'card4')
plt.plot(c4)  

plt.figure(figsize=(8,5))
plt.title('Cumulation of Card5')
c5 = cumulate(train_transaction,'card5')
plt.plot(c5)

plt.figure(figsize=(8,5))
plt.title('Cumulation of Card6')
c6 = cumulate(train_transaction,'card6')
plt.plot(c6)  
  

# Analysing addr1, addr2
plt.figure(figsize=(8,5))
plt.title('Cumulation of addr1')
a1 = cumulate(train_transaction,'addr1')
plt.plot(a1) 

plt.figure(figsize=(8,5))
plt.title('Cumulation of addr2')
a2 = cumulate(train_transaction,'addr2')
plt.plot(a2) 


# Analysing E-MAIL
plt.figure(figsize=(8,5))
plt.title('Cumulation of P_emaildomain')
e1 = cumulate(train_transaction,'P_emaildomain')
plt.plot(e1) 

plt.figure(figsize=(8,5))
plt.title('Cumulation of R_emaildomain')
e2 = cumulate(train_transaction,'R_emaildomain')
plt.plot(e2)


# Analysing M1 ~ M9

M_cat = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
for m in M_cat:
    plt.figure(figsize=(8,5))
    plt.title('Cumulation of ' + m)
    c_m = cumulate(train_transaction, m)
    #plt.plot(c_m)
    #plt.savefig(c_m)



# Removing NaN
def clean(serie):
    output = serie[(np.isnan(serie) == False) & (np.isinf(serie) == False)]
    return output

# Cumulative Distribution Function for continuous feature
def cumulate_continue(data,col):
    print('Class: ', col)
    print('Percentage of Missing Value: '+str(data[col].isnull().sum()/data.shape[0]))
    
    num_category = data[col].value_counts().shape
    print('Number of Categories: '+str(num_category[0]),'\n')
    #print('Category Feature'+'\n'+str(data[col].value_counts()))
    tp=(data[col].value_counts()/data.shape[0]).values.flatten().tolist()
    cumulate=[]
    plt.figure(figsize=(16,10))
    for i in range(1,len(tp)+1):
        cumulate.append(sum(tp[0:i]))
    plt.subplot(221)
    plt.title('Cumulation')
    plt.plot(cumulate)  
    plt.subplot(222)
    plt.title('Kernal Density Estimation')
    sns.kdeplot(data[col])
    plt.subplot(223)
    plt.title('Box Plot')
    sns.boxplot(data[col])
    plt.subplot(224)
    plt.title('Logarithm')
    sns.kdeplot(np.log1p(data[col]))
    plt.suptitle(col)
    plt.savefig(col)


# Analysing dist1, dist2
cumulate_continue(train_transaction,'dist1')
cumulate_continue(train_transaction,'dist2')


# Analysing C1 ~ C14
for i in range(1,15):
    j = str(i)
    c = 'C'+j
    cumulate_continue(train_transaction,c)


# Analysing D1 ~ D15
for i in range(1,16):
    j = str(i)
    d = 'D'+j
    cumulate_continue(train_transaction,d)


# Setting bandwidth in Kernel Density Estimation plot
def cumulate_bw(data,col):
    print('Class: ', col)
    print('Percentage of Missing Value: '+str(data[col].isnull().sum()/data.shape[0]))
    
    num_category = data[col].value_counts().shape
    print('Number of Categories: '+str(num_category[0]),'\n')
    #print('Category Feature'+'\n'+str(data[col].value_counts()))
    tp=(data[col].value_counts()/data.shape[0]).values.flatten().tolist()
    cumulate=[]
    plt.figure(figsize=(16,10))
    for i in range(1,len(tp)+1):
        cumulate.append(sum(tp[0:i]))
    plt.subplot(221)
    plt.title('Cumulation')
    plt.plot(cumulate)  
    plt.subplot(222)
    plt.title('Kernel Density Estimation')
    sns.kdeplot(data[col],bw=0.2)
    plt.subplot(223)
    plt.title('Box Plot')
    sns.boxplot(data[col])
    plt.subplot(224)
    plt.title('Logarithm')
    sns.kdeplot(np.log1p(data[col]),bw=0.2)
    plt.suptitle(col)
    plt.savefig(col)


cumulate_bw(train_transaction,'C12')




# Analysing V
def cumulate_V(data,col):
    print(col ,round(data[col].isnull().sum()/data.shape[0],5))
    
    #num_category = data[col].value_counts().shape
    #print('Number of Categories: '+str(num_category[0]))
    #print('类别特征情况'+'\n'+str(data[col].value_counts()))

print('Percentage of Missing Value: ')
for i in range(1,320):
    j = str(i)
    v = 'V'+j
    cumulate_V(train_transaction,v)


# Analysing id

def cumulate_id(data,col):
    print('\n')
    print('Class: ', col)
    print('Missing Value: ' ,round(data[col].isnull().sum()/data.shape[0],5))   
    num_category = data[col].value_counts().shape
    print('Number of Categories: '+str(num_category[0]))
    print('Categories'+'\n'+str(data[col].value_counts()))




id_cat = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08',
          'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16',
          'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24',
          'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32',
          'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']



for id in id_cat:   
    cumulate_id(train_identity,id)
    

def plot_id(data, col1, col2, col3, col4):
    plt.figure(figsize=(16,10))
    plt.subplot(221)
    plt.title(col1)
    plt.hist(data[col1])  
    plt.subplot(222)
    plt.title(col2)
    plt.hist(data[col2])
    plt.subplot(223)
    plt.title(col3)
    plt.hist(data[col3])
    plt.subplot(224)
    plt.title(col4)
    plt.hist(data[col4])
    plt.suptitle(col1+' ~ '+col4)
    plt.savefig(col1+' to '+col4)
  
    
plot_id(train_identity,'id_01', 'id_02', 'id_03', 'id_04')
plot_id(train_identity,'id_05', 'id_06', 'id_07', 'id_08')
plot_id(train_identity,'id_09', 'id_10', 'id_11', 'id_12')
plot_id(train_identity,'id_13', 'id_14', 'id_21', 'id_22')
plot_id(train_identity,'id_17', 'id_18', 'id_19', 'id_20')
plot_id(train_identity,'id_24', 'id_25', 'id_32')




# Comparing the distribution of TransactionAmt in train data and test data
plt.figure(figsize=(8,5))
plt.title('TransactionAmt')
plt.hist(np.log1p(train_transaction.TransactionAmt),bins=100)
plt.hist(np.log1p(test_transaction.TransactionAmt),bins=100)

corr(train,test,['TransactionAmt'])



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




# merge
train_identity['has_id']=1
train=train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test_identity['had_id']=1
test=test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

# categories in train
cats=['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4',
 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16',
 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24',
 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32',
 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']

# categories in test
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


# Sampling without shuffling
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




# Start Training
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

# Feature importance
feature_importances['average'] = feature_importances[[f'fold_{fold_n+1}' for fold_n in range(5)]].mean(axis = 1)
plt.figure(figsize = (8,10))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average',y='feature')
plt.title('50 TOP feature importance over 5 folds average'.format(folds.n_splits))
plt.savefig('feature importance of Cross-validation')

print('Cross-validation Finished \n')
print('\n')
print('\n')


# Dealing with time
# Deleting TransactionDT, replacing it with the derived features
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

# Feature importance
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

# Feature imoortance
feature_importances['average'] = feature_importances[[f'fold_{fold_n+1}' for fold_n in range(5)]].mean(axis = 1)
plt.figure(figsize = (8,10))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average',y='feature')
plt.title('50 TOP feature importance over 5 folds average'.format(folds.n_splits))
plt.savefig('feature importance After dealing with TransactionAmt')

print('Dealing with TransactionAmt Finished \n')
print('\n')
print('\n')





# Dealing with cards, addrs, and emails
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




# Feature importance
feature_importances['average'] = feature_importances[[f'fold_{fold_n+1}' for fold_n in range(5)]].mean(axis = 1)
plt.figure(figsize = (8,10))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average',y='feature')
plt.title('50 TOP feature importance over 5 folds average'.format(folds.n_splits))
plt.savefig('feature importance After Aggregation')

print('Aggregation Finished \n')
print('\n')
print('\n')


