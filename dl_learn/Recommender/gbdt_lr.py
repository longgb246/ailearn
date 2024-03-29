# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2020/03/25
"""
Usage Of 'gbdt_lr.py' : 
"""

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression


df_train = pd.read_csv(
    r'F:\Codes\tensorflow_practice\recommendation\Basic-DeepFM-model\data\train.csv')
df_test = pd.read_csv(
    r'F:\Codes\tensorflow_practice\recommendation\Basic-DeepFM-model\data\test.csv')

NUMERIC_COLS = [
    "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",
]

print(df_test.head(10))

df_test = df_train[8000:]
df_train = df_train[:8000]

y_train = df_train['target']  # training label
y_test = df_test['target']  # testing label
X_train = df_train[NUMERIC_COLS]  # training dataset
X_test = df_test[NUMERIC_COLS]  # testing dataset

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 64,
    'num_trees': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# number of leaves,will be used in feature transformation
num_leaf = 64

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_train)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict and get data on leaves, training data

y_pred = gbm.predict(X_train, pred_leaf=True)

print(np.array(y_pred).shape)
print(y_pred[0])

print('Writing transformed training data')
transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf],
                                       dtype=np.int64)  # N * num_tress * num_leafs
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    transformed_training_matrix[i][temp] += 1
print(transformed_training_matrix.shape)  # (8000, 6400)

y_pred = gbm.predict(X_test, pred_leaf=True)
print('Writing transformed testing data')
transformed_testing_matrix = np.zeros(
    [len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    transformed_testing_matrix[i][temp] += 1
print(transformed_testing_matrix.shape)  # (8000, 6400)

lm = LogisticRegression(penalty='l2', C=0.05)  # logestic model construction
lm.fit(transformed_training_matrix, y_train)  # fitting the data
# Give the probabilty on each label
y_pred_test = lm.predict_proba(transformed_testing_matrix)
#
y_pred_test = lm.predict(transformed_testing_matrix)
