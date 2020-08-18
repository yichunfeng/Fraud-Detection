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
````
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


