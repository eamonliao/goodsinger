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

x_cols = ["mv_valid_play_count","word_num1_nofans","chat_num_nofans","gender","live_duration","word_num5_nofans"
    ,"song_order_bean","word_num4_nofans","get_bean_nofans","consume_user_count","consume_user_count_nofans"
    ,"sing_like_count","word_num2_nofans","sing_score_mean","sing_score_median","song_key_fans_num","word_num3_nofans"
    ,"fans_count","sing_gift_bean","starlevel","masterpk_num","live_count","real_sing_count","fans_cusum","richlevel"
    ,"user_mv_count","self_mv_count","competitorpk_num","word_num10_nofans","get_bean","word_num7_nofans","song_num"
    ,"sing_gift_coin","nofans_sing_gift_coin","song_order_bean_nofans","word_num8_nofans","nofans_sing_gift_count"
    ,"sing_listen_num","nofans_enternum","word_num6_nofans"]
y_col = 'is_good'

def get_input():
    X, Y, starids=dct.main_proce(x_cols,y_col,is_train=False)
    return X, Y, starids

def load_model():
    os.chdir(MODEL_PATH)
    from sklearn.externals import joblib
    gbdt=joblib.load('best_gbdt_week.model')
    mlp =joblib.load('best_mlp_week.model')
    return gbdt,mlp

def train_score(X_train, Y_train, clf):
    Y_pred=clf.predict(X_train)
    score=accuracy_score(Y_train,Y_pred)
    print('训练集准确率:',score)

def fit_data(X,gbdt,mlp,starids):
    gbdt_prob=gbdt.predict_proba(X)[:,1]
    mlp_prob=mlp.predict_proba(X)[:,1]
    mlp_predict=mlp.predict(X)
    pred_df = list(zip(starids, gbdt_prob, mlp_prob))
    pred_df = pd.DataFrame(pred_df, columns=['starid', 'gbdt_prob', 'mlp_prob'])
    result_list = [str(tuple(item)) for item in pred_df.values.tolist()]
    insert_sql = "insert overwrite table temp.good_singer_pred_week values " + ",".join(result_list)
    dct.spark.sql(insert_sql)
    print('预测结果写入成功')

if __name__=='__main__':
    X, Y, starids = get_input()
    gbdt,mlp = load_model()
    fit_data(X, gbdt, mlp, starids)
    pass
