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

The distributions of TransactionDT in train data and test data seem totally different.

I would like to use kolmogorov-smirnov test to check the distribution.
```python
from scipy.stats import ks_2samp
ks_result = ks_2samp(train_transaction['TransactionDT'].values,test_transaction['TransactionDT'].values)
print('KS Test of TransactionDT in train_trasaction and test_transaction: ',ks_result)
```
The result:
```
KS Test of TransactionDT in train_trasaction and test_transaction:
KstestResult(statistic=1.0, pvalue=0.0)
```
