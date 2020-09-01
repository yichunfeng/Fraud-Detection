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
at your own device.

```
conda env create -f environment.yaml
```

## Overview of dataset

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

The distributions in train and test data:

<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/TransactionAmt%20-%20train%20vs%20test.png" width="500" height="400">

Kolmogorov-smirnov test:

```
Class: TransactionAmt
Kolmogorov-Smirnov test:    KS-stat = 0.017615    p-value = 6.158e-74
```

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


### Analysing V1 ~ V319

```
Percentage of Missing Value: 
V1 0.47293
V2 0.47293
V3 0.47293
V4 0.47293
V5 0.47293
V6 0.47293
V7 0.47293
V8 0.47293
V9 0.47293
V10 0.47293
V11 0.47293
V12 0.12882
V13 0.12882
V14 0.12882
V15 0.12882
V16 0.12882
V17 0.12882
V18 0.12882
V19 0.12882
V20 0.12882
V21 0.12882
V22 0.12882
V23 0.12882
V24 0.12882
V25 0.12882
V26 0.12882
V27 0.12882
V28 0.12882
V29 0.12882
V30 0.12882
V31 0.12882
V32 0.12882
V33 0.12882
V34 0.12882
V35 0.28613
V36 0.28613
V37 0.28613
V38 0.28613
V39 0.28613
V40 0.28613
V41 0.28613
V42 0.28613
V43 0.28613
V44 0.28613
V45 0.28613
V46 0.28613
V47 0.28613
V48 0.28613
V49 0.28613
V50 0.28613
V51 0.28613
V52 0.28613
V53 0.13055
V54 0.13055
V55 0.13055
V56 0.13055
V57 0.13055
V58 0.13055
V59 0.13055
V60 0.13055
V61 0.13055
V62 0.13055
V63 0.13055
V64 0.13055
V65 0.13055
V66 0.13055
V67 0.13055
V68 0.13055
V69 0.13055
V70 0.13055
V71 0.13055
V72 0.13055
V73 0.13055
V74 0.13055
V75 0.15099
V76 0.15099
V77 0.15099
V78 0.15099
V79 0.15099
V80 0.15099
V81 0.15099
V82 0.15099
V83 0.15099
V84 0.15099
V85 0.15099
V86 0.15099
V87 0.15099
V88 0.15099
V89 0.15099
V90 0.15099
V91 0.15099
V92 0.15099
V93 0.15099
V94 0.15099
V95 0.00053
V96 0.00053
V97 0.00053
V98 0.00053
V99 0.00053
V100 0.00053
V101 0.00053
V102 0.00053
V103 0.00053
V104 0.00053
V105 0.00053
V106 0.00053
V107 0.00053
V108 0.00053
V109 0.00053
V110 0.00053
V111 0.00053
V112 0.00053
V113 0.00053
V114 0.00053
V115 0.00053
V116 0.00053
V117 0.00053
V118 0.00053
V119 0.00053
V120 0.00053
V121 0.00053
V122 0.00053
V123 0.00053
V124 0.00053
V125 0.00053
V126 0.00053
V127 0.00053
V128 0.00053
V129 0.00053
V130 0.00053
V131 0.00053
V132 0.00053
V133 0.00053
V134 0.00053
V135 0.00053
V136 0.00053
V137 0.00053
V138 0.86124
V139 0.86124
V140 0.86124
V141 0.86124
V142 0.86124
V143 0.86123
V144 0.86123
V145 0.86123
V146 0.86124
V147 0.86124
V148 0.86124
V149 0.86124
V150 0.86123
V151 0.86123
V152 0.86123
V153 0.86124
V154 0.86124
V155 0.86124
V156 0.86124
V157 0.86124
V158 0.86124
V159 0.86123
V160 0.86123
V161 0.86124
V162 0.86124
V163 0.86124
V164 0.86123
V165 0.86123
V166 0.86123
V167 0.76355
V168 0.76355
V169 0.76324
V170 0.76324
V171 0.76324
V172 0.76355
V173 0.76355
V174 0.76324
V175 0.76324
V176 0.76355
V177 0.76355
V178 0.76355
V179 0.76355
V180 0.76324
V181 0.76355
V182 0.76355
V183 0.76355
V184 0.76324
V185 0.76324
V186 0.76355
V187 0.76355
V188 0.76324
V189 0.76324
V190 0.76355
V191 0.76355
V192 0.76355
V193 0.76355
V194 0.76324
V195 0.76324
V196 0.76355
V197 0.76324
V198 0.76324
V199 0.76355
V200 0.76324
V201 0.76324
V202 0.76355
V203 0.76355
V204 0.76355
V205 0.76355
V206 0.76355
V207 0.76355
V208 0.76324
V209 0.76324
V210 0.76324
V211 0.76355
V212 0.76355
V213 0.76355
V214 0.76355
V215 0.76355
V216 0.76355
V217 0.77913
V218 0.77913
V219 0.77913
V220 0.76053
V221 0.76053
V222 0.76053
V223 0.77913
V224 0.77913
V225 0.77913
V226 0.77913
V227 0.76053
V228 0.77913
V229 0.77913
V230 0.77913
V231 0.77913
V232 0.77913
V233 0.77913
V234 0.76053
V235 0.77913
V236 0.77913
V237 0.77913
V238 0.76053
V239 0.76053
V240 0.77913
V241 0.77913
V242 0.77913
V243 0.77913
V244 0.77913
V245 0.76053
V246 0.77913
V247 0.77913
V248 0.77913
V249 0.77913
V250 0.76053
V251 0.76053
V252 0.77913
V253 0.77913
V254 0.77913
V255 0.76053
V256 0.76053
V257 0.77913
V258 0.77913
V259 0.76053
V260 0.77913
V261 0.77913
V262 0.77913
V263 0.77913
V264 0.77913
V265 0.77913
V266 0.77913
V267 0.77913
V268 0.77913
V269 0.77913
V270 0.76053
V271 0.76053
V272 0.76053
V273 0.77913
V274 0.77913
V275 0.77913
V276 0.77913
V277 0.77913
V278 0.77913
V279 2e-05
V280 2e-05
V281 0.00215
V282 0.00215
V283 0.00215
V284 2e-05
V285 2e-05
V286 2e-05
V287 2e-05
V288 0.00215
V289 0.00215
V290 2e-05
V291 2e-05
V292 2e-05
V293 2e-05
V294 2e-05
V295 2e-05
V296 0.00215
V297 2e-05
V298 2e-05
V299 2e-05
V300 0.00215
V301 0.00215
V302 2e-05
V303 2e-05
V304 2e-05
V305 2e-05
V306 2e-05
V307 2e-05
V308 2e-05
V309 2e-05
V310 2e-05
V311 2e-05
V312 2e-05
V313 0.00215
V314 0.00215
V315 0.00215
V316 2e-05
V317 2e-05
V318 2e-05
V319 2e-05
```

### Analysing id_1 ~ id_38

```
Class:  id_01
Missing Value:  0.0
Number of Categories: 77
Categories
-5.0     82170
 0.0     19555
-10.0    11257
-20.0    11211
-15.0     5674
 
-47.0        1
-54.0        1
-86.0        1
-28.0        1
-57.0        1
Name: id_01, Length: 77, dtype: int64
```

```
Class:  id_02
Missing Value:  0.0233
Number of Categories: 115655
Categories
1102.0      11
696.0       10
1116.0       9
1117.0       9
1120.0       9
            ..
171228.0     1
342457.0     1
128226.0     1
118383.0     1
24576.0      1
Name: id_02, Length: 115655, dtype: int64
```

```
Class:  id_03
Missing Value:  0.54016
Number of Categories: 24
Categories
 0.0     63903
 1.0       863
 3.0       668
 2.0       421
 5.0       109
 4.0       100
 6.0        64
-5.0        33
-6.0        31
-4.0        21
-7.0        21
-10.0       17
-8.0        14
-1.0        12
-2.0        12
-3.0         8
-11.0        6
-9.0         6
 7.0         4
-13.0        3
-12.0        3
 9.0         3
 8.0         1
 10.0        1
Name: id_03, dtype: int64
```

```
Class:  id_04
Missing Value:  0.54016
Number of Categories: 15
Categories
 0.0     65739
-5.0       132
-6.0        98
-8.0        64
-4.0        51
-1.0        43
-11.0       35
-12.0       34
-10.0       30
-9.0        26
-13.0       24
-7.0        21
-2.0        15
-3.0        10
-28.0        2
Name: id_04, dtype: int64
```

<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/id_01%20to%20id_04.png" width="500" height="400">


```
Class:  id_05
Missing Value:  0.05108
Number of Categories: 93
Categories
 0.0     92743
 1.0      8293
 2.0      4937
 3.0      3854
 4.0      2941
 
-35.0        1
-34.0        1
-38.0        1
 48.0        1
-39.0        1
Name: id_05, Length: 93, dtype: int64
```

```
Class:  id_06
Missing Value:  0.05108
Number of Categories: 101
Categories
 0.0     91325
-1.0      4687
-5.0      3849
-6.0      3257
-9.0      2634
 
-93.0        4
-95.0        2
-80.0        2
-99.0        1
-89.0        1
Name: id_06, Length: 101, dtype: int64
```

```
Class:  id_07
Missing Value:  0.96426
Number of Categories: 84
Categories
 0.0     409
 16.0    245
 14.0    228
 12.0    208
 15.0    186

-15.0      1
 51.0      1
 52.0      1
-33.0      1
 61.0      1
Name: id_07, Length: 84, dtype: int64
```

```
Class:  id_08
Missing Value:  0.96426
Number of Categories: 94
Categories
-100.0    500
 0.0      261
-34.0     257
-33.0     209
-32.0     185

-98.0       1
-65.0       1
-90.0       1
-97.0       1
-93.0       1
Name: id_08, Length: 94, dtype: int64
```

<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/id_05%20to%20id_08.png" width="500" height="400">

```
Class:  id_09
Missing Value:  0.48052
Number of Categories: 46
Categories
 0.0     70378
 1.0      1616
 3.0       966
 2.0       773
 4.0       270
 5.0       207
 6.0       145
-6.0        66
-5.0        60
-4.0        42
-10.0       39
-7.0        39
-1.0        38
-8.0        37
 7.0        33
-3.0        31
-2.0        27
-9.0        27
 8.0        23
-11.0       18
 9.0        16
 10.0       11
 11.0        6
 12.0        6
-17.0        4
-12.0        4
-23.0        4
-22.0        4
 13.0        4
-13.0        4
-21.0        3
 16.0        3
 15.0        3
-31.0        3
-19.0        2
-15.0        2
-26.0        2
-18.0        2
 17.0        1
 25.0        1
 14.0        1
-30.0        1
-14.0        1
-24.0        1
-20.0        1
-36.0        1
Name: id_09, dtype: int64
```

```
Class:  id_10
Missing Value:  0.48052
Number of Categories: 62
Categories
 0.0     72879
-6.0       295
-5.0       247
-1.0       200
-8.0       147
 
-58.0        1
-51.0        1
-59.0        1
-50.0        1
-53.0        1
Name: id_10, Length: 62, dtype: int64
```

```
Class:  id_11
Missing Value:  0.02257
Number of Categories: 365
Categories
100.000000    133162
95.080002       1231
95.160004        754
97.120003        440
96.669998        333
 
97.870003          1
97.660004          1
95.040001          1
91.949997          1
94.169998          1
Name: id_11, Length: 365, dtype: int64
```

```
Class:  id_12
Missing Value:  0.0
Number of Categories: 2
Categories
NotFound    123025
Found        21208
Name: id_12, dtype: int64
```

<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/id_09%20to%20id_12.png" width="500" height="400">

```
Class:  id_13
Missing Value:  0.11726
Number of Categories: 54
Categories
52.0    58099
49.0    26365
64.0    14429
33.0    10048
27.0     3666
20.0     2878
14.0     2499
63.0     1468
19.0     1147
25.0     1066
43.0      842
62.0      813
18.0      688
41.0      654
55.0      523
11.0      373
15.0      372
24.0      318
35.0      317
51.0      196
61.0      108
44.0       85
54.0       67
28.0       59
39.0       34
31.0       29
45.0       26
38.0       22
13.0       21
58.0       17
36.0       15
32.0       10
48.0       10
56.0        9
30.0        7
22.0        5
53.0        4
47.0        3
60.0        3
17.0        3
21.0        3
57.0        3
46.0        3
26.0        2
12.0        2
23.0        1
40.0        1
42.0        1
34.0        1
37.0        1
10.0        1
59.0        1
29.0        1
50.0        1
Name: id_13, dtype: int64
```

```
Class:  id_14
Missing Value:  0.44504
Number of Categories: 25
Categories
-300.0    44121
-360.0    16661
-480.0    12891
-420.0     4542
-600.0      498
 60.0       369
 0.0        192
-240.0      159
-180.0      126
-540.0      111
 480.0       80
 540.0       64
 600.0       62
 120.0       41
 180.0       37
 420.0       19
 330.0       17
 270.0       15
 240.0       13
 300.0       12
 720.0        9
-660.0        2
-120.0        1
-210.0        1
 360.0        1
Name: id_14, dtype: int64
```

```
Class:  id_15
Missing Value:  0.02252
Number of Categories: 3
Categories
Found      67728
New        61612
Unknown    11645
Name: id_15, dtype: int64
```

```
Class:  id_16
Missing Value:  0.10326
Number of Categories: 2
Categories
Found       66324
NotFound    63016
Name: id_16, dtype: int64
```

```
Class:  id_17
Missing Value:  0.03372
Number of Categories: 104
Categories
166.0    78631
225.0    56968
102.0      689
159.0      352
100.0      336
 
188.0        1
154.0        1
105.0        1
219.0        1
125.0        1
Name: id_17, Length: 104, dtype: int64
```

```
Class:  id_18
Missing Value:  0.68722
Number of Categories: 18
Categories
15.0    25489
13.0    13439
12.0     4656
18.0      650
20.0      339
17.0      233
26.0       89
21.0       78
24.0       52
11.0       36
27.0       32
29.0        9
23.0        4
14.0        3
16.0        1
28.0        1
10.0        1
25.0        1
Name: id_18, dtype: int64
```

```
Class:  id_19
Missing Value:  0.03408
Number of Categories: 522
Categories
266.0    19849
410.0    11318
427.0     8808
529.0     8122
312.0     6227
 
363.0        1
174.0        1
386.0        1
179.0        1
425.0        1
Name: id_19, Length: 522, dtype: int64
```

```
Class:  id_20
Missing Value:  0.03447
Number of Categories: 394
Categories
507.0    22311
222.0    11065
325.0     8133
533.0     6611
214.0     5664
 
426.0        1
459.0        1
221.0        1
109.0        1
486.0        1
Name: id_20, Length: 394, dtype: int64
```

```
Class:  id_21
Missing Value:  0.96423
Number of Categories: 490
Categories
252.0    2542
228.0     239
255.0     109
596.0     103
576.0     101

431.0       1
648.0       1
635.0       1
637.0       1
540.0       1
Name: id_21, Length: 490, dtype: int64
```

```
Class:  id_22
Missing Value:  0.96416
Number of Categories: 25
Categories
14.0    4736
41.0     321
33.0      38
17.0       7
21.0       7
39.0       6
35.0       5
36.0       5
31.0       5
12.0       5
22.0       5
20.0       4
26.0       4
24.0       4
28.0       3
42.0       3
38.0       2
19.0       2
40.0       1
43.0       1
37.0       1
23.0       1
10.0       1
44.0       1
18.0       1
Name: id_22, dtype: int64
```

```
Class:  id_23
Missing Value:  0.96416
Number of Categories: 3
Categories
IP_PROXY:TRANSPARENT    3489
IP_PROXY:ANONYMOUS      1071
IP_PROXY:HIDDEN          609
Name: id_23, dtype: int64
```

```
Class:  id_24
Missing Value:  0.96709
Number of Categories: 12
Categories
11.0    2817
15.0    1594
16.0     220
18.0      37
21.0      33
24.0      12
17.0       9
26.0       8
25.0       7
19.0       5
12.0       4
23.0       1
Name: id_24, dtype: int64
```

```
Class:  id_25
Missing Value:  0.96442
Number of Categories: 341
Categories
321.0    2494
205.0     301
426.0     236
501.0     103
371.0      83

543.0       1
398.0       1
467.0       1
342.0       1
392.0       1
Name: id_25, Length: 341, dtype: int64
```

```
Class:  id_26
Missing Value:  0.9642
Number of Categories: 95
Categories
161.0    824
184.0    582
142.0    528
102.0    451
100.0    433

101.0      1
198.0      1
178.0      1
115.0      1
127.0      1
Name: id_26, Length: 95, dtype: int64
```

```
Class:  id_27
Missing Value:  0.96416
Number of Categories: 2
Categories
Found       5155
NotFound      14
Name: id_27, dtype: int64
```
```
Class:  id_28
Missing Value:  0.02257
Number of Categories: 2
Categories
Found    76232
New      64746
Name: id_28, dtype: int64
```

```
Class:  id_29
Missing Value:  0.02257
Number of Categories: 2
Categories
Found       74926
NotFound    66052
Name: id_29, dtype: int64
```

```
Class:  id_30
Missing Value:  0.46222
Number of Categories: 75
Categories
Windows 10          21155
Windows 7           13110
iOS 11.2.1           3722
iOS 11.1.2           3699
Android 7.0          2871
 
func                   10
iOS 11.4.0              5
Mac OS X 10_13_5        4
Windows                 3
iOS 11.4.1              1
Name: id_30, Length: 75, dtype: int64
```
```
Class:  id_31
Missing Value:  0.02739
Number of Categories: 130
Categories
chrome 63.0                22000
mobile safari 11.0         13423
mobile safari generic      11474
ie 11.0 for desktop         9030
safari generic              8195
 
Cherry                         1
cyberfox                       1
chrome 67.0 for android        1
iron                           1
seamonkey                      1
Name: id_31, Length: 130, dtype: int64
```

```
Class:  id_32
Missing Value:  0.46208
Number of Categories: 4
Categories
24.0    53071
32.0    24428
16.0       81
0.0         6
Name: id_32, dtype: int64
```

```
Class:  id_33
Missing Value:  0.49187
Number of Categories: 260
Categories
1920x1080    16874
1366x768      8605
1334x750      6447
2208x1242     4900
1440x900      4384
 
2559x1440        1
1188x720         1
1264x924         1
1440x803         1
1280x740         1
Name: id_33, Length: 260, dtype: int64
```

```
Class:  id_34
Missing Value:  0.46056
Number of Categories: 4
Categories
match_status:2     60011
match_status:1     17376
match_status:0       415
match_status:-1        3
Name: id_34, dtype: int64
```

```
Class:  id_35
Missing Value:  0.02252
Number of Categories: 2
Categories
T    77814
F    63171
Name: id_35, dtype: int64
```

```
Class:  id_36
Missing Value:  0.02252
Number of Categories: 2
Categories
F    134066
T      6919
Name: id_36, dtype: int64
```

```
Class:  id_37
Missing Value:  0.02252
Number of Categories: 2
Categories
T    110452
F     30533
Name: id_37, dtype: int64
```

```
Class:  id_38
Missing Value:  0.02252
Number of Categories: 2
Categories
F    73922
T    67063
Name: id_38, dtype: int64
```

```
Class:  DeviceType
Missing Value:  0.02373
Number of Categories: 2
Categories
desktop    85165
mobile     55645
Name: DeviceType, dtype: int64
```

```
Class:  DeviceInfo
Missing Value:  0.17726
Number of Categories: 1786
Categories
Windows        47722
iOS Device     19782
MacOS          12573
Trident/7.0     7440
rv:11.0         1901
 
STV100-3           1
SM-G7105           1
KYY22              1
LGL33L/V100        1
A0001              1
Name: DeviceInfo, Length: 1786, dtype: int64
```

## Preprocessing

Merging the transaction and identity data:

```python
train_identity['has_id']=1
train=train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test_identity['had_id']=1
test=test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
```

Here I use the function below to reduce the memory usage in pandas:

```python
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
```

Sampling without shuffling:

```python
X = copy.deepcopy(train)
X_test = copy.deepcopy(test)
```

Setting the target:
```python
y = X["isFraud"]
del X["isFraud"]

print("X shape:{}, X_test shape:{}".format(X.shape, X_test.shape))
print("y shape:{}".format(y.shape))
```

```
Finished sampling: 
	X shape:(590540, 433), X_test shape:(506691, 433)
	y shape:(590540,) 
```


## Feature Engineering

According to the organizer, once a user is defined as 'Fraud', then all of his corresponding transactions will be marked as 'Fraud'.
However, how do we determine that multiple transactions that come from the same user? The host stated the features below:

1. The features of email addresses P and R email

2. The address, which is obviously the feature of addr1 and addr2

3. User account

Here I use the negative sampling and observe its unique value to roughly determine whether this feature can be used for uniqueness determination:

```python
X['y']=y
fraud=X[X.y==1]
```

Checking the missing value of card1 ~ card6 to determine whether they can be mapped to unique users:

```python
card1 = fraud.card1.value_counts()
print('number of missing value in card1: ',fraud.card1.isnull().sum())
card1 = card1[card1>0]
print('number of unique card1: ',card1.shape[0])
```

```
number of missing value in card1:  0
number of unique card1:  1740
number of missing value in card2:  0
number of unique card2:  328
number of missing value in card3:  0
number of unique card3:  63
number of missing value in card4:  0
number of unique card4:  5
number of missing value in card5:  0
number of unique card5:  50
number of missing value in card6:  0
number of unique card6:  3 
```

There are no missing values in all cards, so we can determine the unique card through card1 - card6.
Later I would like to aggregate features of card1 to card6.

Observing the address:

```
number of missing value in addr1:  7741
number of missing value in addr2:  7741 
```
It might be that the same users lose both addr1 and addr2.

Observing the email:

```
number of missing value in P_emaildomain:  0
number of missing value in R_emaildomain:  0
```

The aggregation of the above features will be added in the training later to see whether the score can be improved.


Doing the cross-validation to see which feature is the most important:
```python
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
```

```
Training until validation scores don't improve for 100 rounds
[200]	training's auc: 1	valid_1's auc: 0.880561
[400]	training's auc: 1	valid_1's auc: 0.885042
Early stopping, best iteration is:
[336]	training's auc: 1	valid_1's auc: 0.888068
Fold 1 | AUC: 0.8880677190000972
Training until validation scores don't improve for 100 rounds
Early stopping, best iteration is:
[35]	training's auc: 0.997641	valid_1's auc: 0.904081
Fold 2 | AUC: 0.9040812272361682
Training until validation scores don't improve for 100 rounds
Early stopping, best iteration is:
[60]	training's auc: 0.999918	valid_1's auc: 0.919139
Fold 3 | AUC: 0.9191391794238797
Training until validation scores don't improve for 100 rounds
[200]	training's auc: 1	valid_1's auc: 0.950676
[400]	training's auc: 1	valid_1's auc: 0.951627
[600]	training's auc: 1	valid_1's auc: 0.95223
Early stopping, best iteration is:
[521]	training's auc: 1	valid_1's auc: 0.952742
Fold 4 | AUC: 0.9527418489851609
Training until validation scores don't improve for 100 rounds
[200]	training's auc: 1	valid_1's auc: 0.919926
[400]	training's auc: 1	valid_1's auc: 0.923399
Early stopping, best iteration is:
[359]	training's auc: 1	valid_1's auc: 0.923997
Fold 5 | AUC: 0.9239973279337228

Mean AUC = 0.9176054605158057
Out of folds AUC = 0.7895479686846192
```

The feature importance:
```python
feature_importances['average'] = feature_importances[[f'fold_{fold_n+1}' for fold_n in range(5)]].mean(axis = 1)
plt.figure(figsize = (8,10))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average',y='feature')
plt.title('50 TOP feature importance over 5 folds average'.format(folds.n_splits))
plt.savefig('feature importance of Cross-validation')
```

<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/Feature%20Importance%20-%20Cross-validation.png" width="500" height="400">


TransactionDT is the most important, but there might be an over-fitting outcome, so I try to delete this feature and replace it with
its derived features.
```python
X.drop(['year','month'],axis=1,inplace=True)
X_test.drop(['year','month'],axis=1,inplace=True)
del X['TransactionDT']
del X_test['TransactionDT']
```

Training again:
```
After dealing with TransactionDT...
Training until validation scores don't improve for 100 rounds
[200]	training's auc: 1	valid_1's auc: 0.89926
[400]	training's auc: 1	valid_1's auc: 0.903266
Early stopping, best iteration is:
[401]	training's auc: 1	valid_1's auc: 0.903393
Fold 1 | AUC: 0.9033936132577562
Training until validation scores don't improve for 100 rounds
[200]	training's auc: 1	valid_1's auc: 0.928316
Early stopping, best iteration is:
[153]	training's auc: 1	valid_1's auc: 0.928875
Fold 2 | AUC: 0.9288750375327399
Training until validation scores don't improve for 100 rounds
[200]	training's auc: 1	valid_1's auc: 0.925962
[400]	training's auc: 1	valid_1's auc: 0.928101
Early stopping, best iteration is:
[446]	training's auc: 1	valid_1's auc: 0.928623
Fold 3 | AUC: 0.9286232994223065
Training until validation scores don't improve for 100 rounds
[200]	training's auc: 1	valid_1's auc: 0.948452
[400]	training's auc: 1	valid_1's auc: 0.949067
Early stopping, best iteration is:
[363]	training's auc: 1	valid_1's auc: 0.949294
Fold 4 | AUC: 0.949294471649418
Training until validation scores don't improve for 100 rounds
[200]	training's auc: 1	valid_1's auc: 0.920547
[400]	training's auc: 1	valid_1's auc: 0.920644
Early stopping, best iteration is:
[352]	training's auc: 1	valid_1's auc: 0.921646
Fold 5 | AUC: 0.9216464084861615

Mean AUC = 0.9263665660696764
Out of folds AUC = 0.9096124143011752
```

Feature Importance after dealing with TransactionDT:

<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/Feature%20Importance%20-%20TransactionDT.png" width="500" height="400">


Observing the PSI:

```
PSI:              feature    fold_1    fold_2    fold_3    fold_4    fold_5
0     TransactionDT  0.000531  0.000926  0.000569  0.000220  0.000018
1    TransactionAmt  0.760167  0.218534  0.240927  0.225956  0.270768
2             dist1  0.116246  0.033917  0.038364  0.034633  0.049485
3             dist2  0.034019  0.019909  0.026873  0.027001  0.034547
4                C1  0.066917  0.012865  0.010096  0.013650  0.018613
..              ...       ...       ...       ...       ...       ...
379           id_08  0.011754  0.002461  0.002882  0.003631  0.002048
380           id_09  0.124157  0.010760  0.018326  0.012398  0.004354
381           id_10  0.123148  0.011459  0.018413  0.012261  0.003649
382           id_11  0.309666  0.017178  0.044611  0.024725  0.018522
383          has_id  0.313582  0.016269  0.044783  0.022193  0.016269
```

Then, I would like to process TransactionAmt.
Trying the binning for TransactionAmt:

```python
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
```

Result:

```
Class: TransactionAmt
Original Information Value:  0.2058847344507144
Original Population Stability Index:  0.34327045481107543

IV after cart tree:  [0.12198830816553775, 0.2058847344507144, 0.2058847344507144, 0.2058847344507144, 0.2058847344507144, 0.2058847344507144, 0.2058847344507144, 0.2058847344507144, 0.2058847344507144, 0.2058847344507144]
PSI after cart tree:  [0.0012451352953930163, 0.004525429269179209, 0.0051395623311646784, 0.008239099643990326, 0.014556160598400153, 0.02739766926485464, 0.028261124134124783, 0.029407827402452253, 0.030342983010878687, 0.03138445353451209]

IV after chimerge:  [0.12553957942969127, 0.22327379793772928, 0.2317534180526106, 0.19837567666518288, 0.19837567666518288, 0.2049864218284863, 0.2068895507881729, 0.2068895507881729, 0.2068895507881729, 0.20470628154222392]
PSI after chimerge:  [0.001481575351249291, 0.017597195224925075, 0.028592359913524046, 0.0341737284431819, 0.04163968550182455, 0.04353819303092197, 0.06317425179249572, 0.06492120435096087, 0.07434188608361532, 0.08047228260474705]

IV after kmeans merge:  [0.022056371653563604, 0.0315928849560944, 0.0621661699657942, 0.07934223936302433, 0.10067971618971677, 0.12544475947867806, 0.11972008501933762, 0.14955465067581158, 0.14757507259257743, 0.15062661292198293]
PSI after kmeans merge:  [0.0008498433829856464, 0.0023143436118735856, 0.003915125844458691, 0.0034460363712985275, 0.004657652179627167, 0.0073015854370605355, 0.019002772221801127, 0.03534854319035138, 0.03271971388528722, 0.03415572679722579]
```

The effect is best when using cart tree with bins = 10, but the training result is not as expected. The magical treatment for TransactionAmt provided in the  Kaggle discussion: the decimal point of the transaction amount determines countries! Therefore, I deprecate binning.


```python
X['TransactionAmt_decimal'] = ((X['TransactionAmt'] - X['TransactionAmt'].astype(int)) * 1000).astype(int)
X['TransactionAmt'] = X['TransactionAmt_decimal']
del X['TransactionAmt_decimal']
```

Training:

```
Training until validation scores don't improve for 100 rounds
[200]	training's auc: 1	valid_1's auc: 0.89428
Early stopping, best iteration is:
[279]	training's auc: 1	valid_1's auc: 0.895762
Fold 1 | AUC: 0.8957618775875847
Training until validation scores don't improve for 100 rounds
[200]	training's auc: 1	valid_1's auc: 0.924157
Early stopping, best iteration is:
[202]	training's auc: 1	valid_1's auc: 0.92421
Fold 2 | AUC: 0.9242097600106162
Training until validation scores don't improve for 100 rounds
[200]	training's auc: 1	valid_1's auc: 0.922459
[400]	training's auc: 1	valid_1's auc: 0.92396
Early stopping, best iteration is:
[381]	training's auc: 1	valid_1's auc: 0.924385
Fold 3 | AUC: 0.9243850758082311
Training until validation scores don't improve for 100 rounds
[200]	training's auc: 1	valid_1's auc: 0.946515
Early stopping, best iteration is:
[296]	training's auc: 1	valid_1's auc: 0.947904
Fold 4 | AUC: 0.9479037128442651
Training until validation scores don't improve for 100 rounds
[200]	training's auc: 1	valid_1's auc: 0.916775
Early stopping, best iteration is:
[243]	training's auc: 1	valid_1's auc: 0.917828
Fold 5 | AUC: 0.9178278693201013

Mean AUC = 0.9220176591141597
Out of folds AUC = 0.9164658130152636
```
The AUC score has improved.

Feature Importance after dealing with TransactionAmt:

<img src="https://github.com/yichunfeng/Fraud-Detection/blob/master/Figure/Feature%20Importance%20-%20TransactionAmt.png" width="500" height="400">

## Training

Finally adding the derived feature from Aggregation:

```python
X['identity']=X.card1.astype(str)+'_'+X.card2.astype(str)+'_'+X.card3.astype(str)+'_'+ \
X.card4.astype(str)+'_'+X.card5.astype(str)+'_'+X.card6.astype(str)

X.identity=X.identity.astype(str)+'_'+X.addr1.astype(str)+'_'+X.addr2.astype(str)+'_'+X.P_emaildomain.astype(str)+'_'+X.R_emaildomain.astype(str)
X.identity=X.identity.astype('category')
```

And the final parameter:

```python
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
```

```
Aggregation of card, addr and email...
Training until validation scores don't improve for 100 rounds
[200]	training's auc: 0.979597	valid_1's auc: 0.89193
[400]	training's auc: 0.989708	valid_1's auc: 0.89585
[600]	training's auc: 0.995078	valid_1's auc: 0.898025
[800]	training's auc: 0.997739	valid_1's auc: 0.898831
[1000]	training's auc: 0.998996	valid_1's auc: 0.898914
Did not meet early stopping. Best iteration is:
[1000]	training's auc: 0.998996	valid_1's auc: 0.898914
Fold 1 | AUC: 0.8989143877948931
Training until validation scores don't improve for 100 rounds
[200]	training's auc: 0.978032	valid_1's auc: 0.924345
[400]	training's auc: 0.989142	valid_1's auc: 0.929785
[600]	training's auc: 0.994865	valid_1's auc: 0.931901
Early stopping, best iteration is:
[585]	training's auc: 0.994581	valid_1's auc: 0.932046
Fold 2 | AUC: 0.9320463310969629
Training until validation scores don't improve for 100 rounds
[200]	training's auc: 0.978493	valid_1's auc: 0.921931
[400]	training's auc: 0.989409	valid_1's auc: 0.927512
[600]	training's auc: 0.995022	valid_1's auc: 0.930028
[800]	training's auc: 0.997701	valid_1's auc: 0.930296
Early stopping, best iteration is:
[739]	training's auc: 0.997076	valid_1's auc: 0.930553
Fold 3 | AUC: 0.9305532522036665
Training until validation scores don't improve for 100 rounds
[200]	training's auc: 0.97831	valid_1's auc: 0.938248
[400]	training's auc: 0.989269	valid_1's auc: 0.944985
[600]	training's auc: 0.994733	valid_1's auc: 0.9484
[800]	training's auc: 0.997468	valid_1's auc: 0.949629
[1000]	training's auc: 0.998828	valid_1's auc: 0.949777
Did not meet early stopping. Best iteration is:
[1000]	training's auc: 0.998828	valid_1's auc: 0.949777
Fold 4 | AUC: 0.9497765750731585
Training until validation scores don't improve for 100 rounds
[200]	training's auc: 0.978412	valid_1's auc: 0.913059
[400]	training's auc: 0.989309	valid_1's auc: 0.919806
[600]	training's auc: 0.994846	valid_1's auc: 0.922772
[800]	training's auc: 0.997586	valid_1's auc: 0.923509
Early stopping, best iteration is:
[766]	training's auc: 0.99725	valid_1's auc: 0.923626
Fold 5 | AUC: 0.9236257861082804

Mean AUC = 0.9269832664553922
Out of folds AUC = 0.9257745501860899
```

## Author
Yi-Chun, Feng
