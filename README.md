# IEEE-CIS Fraud Detection

## Dataset

[Input](https://www.kaggle.com/c/ieee-fraud-detection/data)

## Requirements

* Python 3.6+
* pandas
* numpy
* sklearn
* matplotlib
* seaborn
* lightgbm
* toad
* category_encoders
* scipy

For Anaconda users, you can simply load *env_fraud_detection.yaml* and create the same environment
at you own device.

```
conda env create -f environment.yaml
```

## Overview of train_transaciton

### Categorical features - Transaction
* ProductCD
* card1 - card6
* addr1, addr2
* P_emaildomain
* R_emaildomain
* M1 - M9
### Categorical features - Identity
* DeviceType
* DeviceInfo
* id_12 - id_38

### Analysing the time consistency

<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/TransactionDT%20in%20train_transaction.png" width="500" height="400">

<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/TransactionDT%20in%20test_transaction.png" width="500" height="400">

The distributions of TransactionDT in train data and test data seem different.

I would like to use kolmogorov-smirnov test to check the distribution.
```python
from scipy.stats import ks_2samp
ks_result = ks_2samp(train_transaction['TransactionDT'].values,test_transaction['TransactionDT'].values)
print('KS Test of TransactionDT in train_trasaction and test_transaction: ',ks_result)
```
The result idicates that the distributions are totally different:
```
KS Test of TransactionDT in train_trasaction and test_transaction:
KstestResult(statistic=1.0, pvalue=0.0)
```

According to the competition host, the period of time is important. If one transation is not labelled as 'Fraud' for over 120 days, then this transaction
is regarded as a valid one. Therefore, TransactionDT needs to be transformed. 

```python
train=pd.DataFrame()
train['time'] = pd.to_datetime(train_transaction['TransactionDT']+1546272000, unit='s')
train['year'] = train["time"].dt.year
train['month'] = train["time"].dt.month
train['day'] = train["time"].dt.day
train['hour']  = train["time"].dt.hour
train['minute']  = train["time"].dt.minute
train['weekday']  = train["time"].dt.dayofweek
train.pop('time')

test=pd.DataFrame()
test['time'] = pd.to_datetime(test_transaction['TransactionDT']+1546272000, unit='s')
test['year'] = test["time"].dt.year
test['month'] = test["time"].dt.month
test['day'] = test["time"].dt.day
test['hour']  = test["time"].dt.hour
test['minute']  = test["time"].dt.minute
test['weekday']  = test["time"].dt.dayofweek
test.pop('time')
```
Doing the kolmogorov-smirnov test again:
```python
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
```
The results:
```
 Class: year
 Kolmogorov-Smirnov test:    KS-stat = 0.235465    p-value = 0.000e+00

 Class: month
 Kolmogorov-Smirnov test:    KS-stat = 0.764535    p-value = 0.000e+00

 Class: day
 Kolmogorov-Smirnov test:    KS-stat = 0.025674    p-value = 1.368e-156

 Class: hour
 Kolmogorov-Smirnov test:    KS-stat = 0.017204    p-value = 1.529e-70

 Class: minute
 Kolmogorov-Smirnov test:    KS-stat = 0.002595    p-value = 5.066e-02

 Class: weekday
 Kolmogorov-Smirnov test:    KS-stat = 0.016545    p-value = 2.826e-65
```
For 'year' and 'month':
```python
print('year in train: ',train.year.value_counts())
print('year in test: ',test.year.value_counts())
print('month in train: ',train.month.value_counts())
print('month in test: ', test.month.value_counts())
```
The results:
```
year in train:  2019    590540
Name: year, dtype: int64

year in test:  2019    387383
2020    119308
Name: year, dtype: int64

month in train:  1    135222
4     97878
3     94467
5     87385
6     86906
2     83822
7      4860
Name: month, dtype: int64

month in test:  1     119308
12     84685
11     77510
8      76379
10     75072
9      73737
Name: month, dtype: int64
```
TransactionDTs in train data and the test data only have the "January" overlap, and the others are in different ranges. Those useless features might 
be deleted later while using LightGBM for training.

### Analysing the TransactionAmt
Kernel density estimation:
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/TransactionAmt%20in%20train_transaction.png" width="500" height="400">
Kernel density estimation after Logarithmic transformation：
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/TransactionAmt%20in%20train_transaction%20-%20Logarithm.png" width="500" height="400">
Box plot of TransactionAmt:
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/TransactionAmt%20in%20train_transaction%20-%20Box%20Plot.png" width="500" height="400">
There exist some extreme values.

### Analysing productCD
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/Count%20of%20Observations%20by%20ProductCD.png" width="500" height="400">
```
Number of Missing Values in ProductCD in train_transaction: 0
```
### Analysing card1 ~ card6
Observing the missing values and cumulative distribution function:
```
Class: card1
Percentage of Missing Value: 0.0
Number of Categories: 13553 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/Cumulation%20of%20card1.png" width="500" height="400">
```
Class: card2
Percentage of Missing Value: 0.015126833068039422
Number of Categories: 500 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/Cumulation%20of%20card2.png" width="500" height="400">
```
Class: card3
Percentage of Missing Value: 0.0026501168422122124
Number of Categories: 114 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/Cumulation%20of%20card3.png" width="500" height="400">
```
Class: card4
Percentage of Missing Value: 0.00267043722694483
Number of Categories: 4 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/Cumulation%20of%20card4.png" width="500" height="400">
```
Class: card5
Percentage of Missing Value: 0.007212043214684865
Number of Categories: 119 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/Cumulation%20of%20card5.png" width="500" height="400">
```
Class: card6
Percentage of Missing Value: 0.0026602770345785214
Number of Categories: 4 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/Cumulation%20of%20card6.png" width="500" height="400">

### Analysing addr1 and addr2
Observing the missing values and cumulative distribution function:
```
Class: addr1
Percentage of Missing Value: 0.1112642666034477
Number of Categories: 332 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/Cumulation%20of%20addr1.png" width="500" height="400">
```
Class: addr2
Percentage of Missing Value: 0.1112642666034477
Number of Categories: 74 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/Cumulation%20of%20addr2.png" width="500" height="400">

### Analysing P_emaildomain and R_emaildomain
Observing the missing values and cumulative distribution function:
```
Class: P_emaildomain
Percentage of Missing Value: 0.1599485216920107
Number of Categories: 59 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/Cumulation%20of%20P_email.png" width="500" height="400">
```
Class: R_emaildomain
Percentage of Missing Value: 0.7675161716395164
Number of Categories: 60 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/Cumulation%20of%20R_email.png" width="500" height="400">

### Analysing M1 ~ M9
```
Class: M1
Percentage of Missing Value: 0.4590713584177194
Number of Categories: 2 

Class: M2
Percentage of Missing Value: 0.4590713584177194
Number of Categories: 2 

Class: M3
Percentage of Missing Value: 0.4590713584177194
Number of Categories: 2 

Class: M4
Percentage of Missing Value: 0.47658753005723575
Number of Categories: 3 

Class: M5
Percentage of Missing Value: 0.5934940901547736
Number of Categories: 2 

Class: M6
Percentage of Missing Value: 0.28678836319300977
Number of Categories: 2 

Class: M7
Percentage of Missing Value: 0.5863531682866528
Number of Categories: 2 

Class: M8
Percentage of Missing Value: 0.5863311545365258
Number of Categories: 2 

Class: M9
Percentage of Missing Value: 0.5863311545365258
Number of Categories: 2 
```

### Analysing dist1 and dist2
```python
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
```

```
Class:  dist1
Percentage of Missing Value: 0.596523520845328
Number of Categories: 2651 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/dist1.png" width="500" height="400">

```
Class:  dist2
Percentage of Missing Value: 0.9362837403054831
Number of Categories: 1751 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/dist2.png" width="500" height="400">

### Analysing C1 ~ C14
```
Class:  C1
Percentage of Missing Value: 0.0
Number of Categories: 1657 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/C1.png" width="500" height="400">
```
Class:  C2
Percentage of Missing Value: 0.0
Number of Categories: 1216 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/C2.png" width="500" height="400">
```
Class:  C3
Percentage of Missing Value: 0.0
Number of Categories: 27 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/C3.png" width="500" height="400">
```
Class:  C4
Percentage of Missing Value: 0.0
Number of Categories: 1260 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/C4.png" width="500" height="400">
```
Class:  C5
Percentage of Missing Value: 0.0
Number of Categories: 319 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/C5.png" width="500" height="400">
```
Class:  C6
Percentage of Missing Value: 0.0
Number of Categories: 1328 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/C6.png" width="500" height="400">
```
Class:  C7
Percentage of Missing Value: 0.0
Number of Categories: 1103 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/C7.png" width="500" height="400">
```
Class:  C8
Percentage of Missing Value: 0.0
Number of Categories: 1253 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/C8.png" width="500" height="400">
```
Class:  C9
Percentage of Missing Value: 0.0
Number of Categories: 205 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/C9.png" width="500" height="400">
```
Class:  C10
Percentage of Missing Value: 0.0
Number of Categories: 1231 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/C10.png" width="500" height="400">
```
Class:  C11
Percentage of Missing Value: 0.0
Number of Categories: 1476 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/C11.png" width="500" height="400">
```
Class:  C12
Percentage of Missing Value: 0.0
Number of Categories: 1199 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/C12.png" width="500" height="400">
```
Class:  C13
Percentage of Missing Value: 0.0
Number of Categories: 1597 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/C13.png" width="500" height="400">
```
Class:  C14
Percentage of Missing Value: 0.0
Number of Categories: 1108 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/C14.png" width="500" height="400">

### Analysing D1 ~ D15
```
Class:  D1
Percentage of Missing Value: 0.0021488806854743116
Number of Categories: 641 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/D1.png" width="500" height="400">
```
Class:  D2
Percentage of Missing Value: 0.4754919226470688
Number of Categories: 641 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/D2.png" width="500" height="400">
```
Class:  D3
Percentage of Missing Value: 0.44514850814508755
Number of Categories: 649 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/D3.png" width="500" height="400">
```
Class:  D4
Percentage of Missing Value: 0.2860466691502693
Number of Categories: 808 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/D4.png" width="500" height="400">
```
Class:  D5
Percentage of Missing Value: 0.524674027161581
Number of Categories: 688 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/D5.png" width="500" height="400">
```
Class:  D6
Percentage of Missing Value: 0.8760676668811597
Number of Categories: 829 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/D6.png" width="500" height="400">
```
Class:  D7
Percentage of Missing Value: 0.9340992989467267
Number of Categories: 597 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/D7.png" width="500" height="400">
```
Class:  D8
Percentage of Missing Value: 0.8731229044603245
Number of Categories: 12353 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/D8.png" width="500" height="400">
```
Class:  D9
Percentage of Missing Value: 0.8731229044603245
Number of Categories: 24 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/D9.png" width="500" height="400">
```
Class:  D10
Percentage of Missing Value: 0.1287330240119213
Number of Categories: 818 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/D10.png" width="500" height="400">
```
Class:  D11
Percentage of Missing Value: 0.47293494090154775
Number of Categories: 676 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/D11.png" width="500" height="400">
```
Class:  D12
Percentage of Missing Value: 0.8904104717715988
Number of Categories: 635 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/D12.png" width="500" height="400">
```
Class:  D13
Percentage of Missing Value: 0.8950926270870728
Number of Categories: 577 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/D13.png" width="500" height="400">
```
Class:  D14
Percentage of Missing Value: 0.8946946862193924
Number of Categories: 802 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/D14.png" width="500" height="400">
```
Class:  D15
Percentage of Missing Value: 0.1509008703898127
Number of Categories: 859 
```
<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/D15.png" width="500" height="400">
