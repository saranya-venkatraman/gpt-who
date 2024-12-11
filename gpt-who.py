# This script takes as input two .csv files with UID features obtained from "get_uid_features.py"
# corresponding to the train and test split of the dataset,
# calculates the UID span features to concatenate with the other 4 (uid_var, 
# uid_diff, uid_diff2 and mean), runs logistic regression, predicts labels,
# and reports performance

import pandas as pd
import random
import numpy as np
from collections import Counter
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from collections import Counter
from functools import reduce
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, default = "./scores/train_uid_scores.csv")
parser.add_argument('--test_file', type=str, default = "./scores/test_uid_scores.csv")

args = parser.parse_args()

train = pd.read_csv(args.train_file)
test= pd.read_csv(args.test_file)

print(len(train), len(test))
k = Counter(train['label'])
label_list = list(k.keys())

    
def transform_probs(df):
    df['probs'] = df['surps']
    df['probs'] = df['probs'].apply(lambda x: re.sub(' +', ',', x))
    df['probs'] = df['probs'].apply(lambda x: re.sub('\n', '', x))
    df['probs'] = df['probs'].apply(lambda x: x.strip('][').split(','))
    df['probs'] = df['probs'].apply(lambda x: list(filter(None, x)))
    df['probs'] = df['probs'].apply(lambda x: list(map(float, x)))
    return df['probs']

# Convert raw surprisal values stored from .csv into integer format 
train['surps'] = transform_probs(train)
test['surps'] = transform_probs(test)

# Calculated UID span features
def spans(lst, n=50):
    """Yield successive n-sized spans from lst."""
    remove_list = []
    max_uid, min_uid = -1, 10000
    span_max, span_min = [], []
    if len(lst) <= n:
        return None
    for i in range(0, len(lst), n):
        span = lst[i:i + n]
        if len(span) == n:
            uid = np.var(span)
            if uid > max_uid:
                max_uid = uid
                span_max = span
            if uid < min_uid:
                min_uid = uid
                span_min = span
    return span_min+span_max    


# Calculate minimum and maximum UID spans in "spans' column
train['spans'] = list(map(spans, train['surps']))
test['spans'] = list(map(spans, test['surps']))
train = train[train['spans'].notna()].reset_index()
test = test[test['spans'].notna()].reset_index()

# Concatenate UID span features with 4 main UID features
features = ['uid_var', 'uid_diff', 'uid_diff2', 'mean']
X_1 = train[features]
X_2 = pd.DataFrame(np.stack(train['spans']))
X_train = pd.concat((X_1, X_2), axis = 1, ignore_index=True)

Z_1 = test[features]
Z_2 = pd.DataFrame(np.stack(test['spans']))
X_test = pd.concat((Z_1, Z_2), axis = 1, ignore_index=True)

y_train = train['label']
y_test = test['label']

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report as cr

model = LogisticRegression(max_iter = 10000)

model.fit(X_train, y_train)
pred = model.predict(X_test)
print(cr(y_test, pred))


