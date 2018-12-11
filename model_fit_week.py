import numpy as np
import os
import pandas as pd
from sklearn.metrics import *
import itertools
import math
import json

FATHER_PATH=os.path.dirname(os.getcwd())
DATA_PATH=os.path.join(FATHER_PATH,'dataset')
MODEL_PATH=os.path.join(FATHER_PATH,'models')
OUT_PATH=os.path.join(FATHER_PATH,'out_put')
LOGGING_PATH=os.path.join(FATHER_PATH,'loggings')
PILOT_PATH=os.path.join(FATHER_PATH,'pilot_data')
EVA_PATH=os.path.join(FATHER_PATH,'model_evaluate')

import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"

os.chdir(LOGGING_PATH)
logging.basicConfig(filename='fit.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

#中文示例
# plt.rcParams['font.sans-serif']=['SimHei']
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('precision', 2)

import data_clean_trans_week as dct

x_cols = ['gender', 'real_sing_count', 'song_num', 'sing_score_mean', 'sing_like_count', 'user_mv_num',
          'word1_rate', 'word4_rate','word7_rate', 'song_order_user_num', 'song_order_sing_rate', 'like_num_per_song',
          'sing_count_per_hour', 'key_fans_rate']
y_col = 'is_good'

def get_input():
    X_train, Y_train, X, starids = dct.main_proce_for_fit(x_cols,y_col)
    return X_train, Y_train, X, starids

def load_model():
    os.chdir(MODEL_PATH)
    from sklearn.externals import joblib
    gbdt=joblib.load('best_gbdt.model')
    mlp =joblib.load('best_mlp.model')
    return gbdt,mlp

def train_score(X_train, Y_train, clf):
    Y_pred=clf.predict(X_train)
    score=accuracy_score(Y_train,Y_pred)
    print('训练集准确率:',score)

def fit_data(X,gbdt,mlp,starids):
    gbdt_prob=gbdt.predict_proba(X)[:,1]
    mlp_prob =mlp.predict_proba(X)[:,1]
    pred_df=list(zip(starids,gbdt_prob,mlp_prob))
    pred_df=pd.DataFrame(pred_df,columns=['starid','gbdt_prob','mlp_prob'])
    result_list = [str(tuple(item)) for item in pred_df.values.tolist()]
    insert_sql = "insert overwrite table temp.good_singer_pred_week values " + ",".join(result_list)
    dct.spark.sql(insert_sql)
    print('预测结果写入成功')

if __name__=='__main__':
    X_train, Y_train, X, starids = get_input()
    gbdt,mlp = load_model()
    train_score(X_train, Y_train, gbdt)
    train_score(X_train, Y_train, mlp)
    fit_data(X, gbdt, mlp, starids)
    pass
