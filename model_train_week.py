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
logging.basicConfig(filename='train.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

#中文示例
# plt.rcParams['font.sans-serif']=['SimHei']
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('precision', 2)

import warnings
warnings.filterwarnings("ignore")

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
    X_data,Y_data,starids=dct.main_proce(x_cols,y_col)
    return X_data,Y_data,starids

def split_data(X_data,Y_data,starids,test_size=0.1):
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test,starids_train,starids_test=train_test_split(X_data,Y_data,starids,test_size=test_size)
    return X_train,X_test,Y_train,Y_test,starids_train,starids_test

def ana_model(X_test,y_true,y_prob,starids):
    df=pd.DataFrame(X_test)
    df.columns=x_cols
    df['y_true']=list(y_true)
    df['y_prob']=y_prob[:,1]
    df['starid']=list(starids)
    df['distance']=(df['y_true']-df['y_prob']).apply(abs)
    df=df[df['distance']>=0.5]
    # print(df)
    os.chdir(OUT_PATH)
    df.to_csv('test_out.csv',sep=',',index=None)

def evaluate_model(y_true,y_prob,thresh=0.5,model_name=None):
    source_dir=os.getcwd()
    os.chdir(EVA_PATH)
    try:
        y_prob=y_prob[:,1]
    except:
        pass
    y_pred=list(map(lambda x:1 if x>thresh else 0,y_prob))
    # 精确率和召回率
    from sklearn.metrics import precision_recall_curve,precision_score,recall_score,f1_score
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_s=f1_score(y_true,y_pred,pos_label=1)
    pre_s=precision_score(y_true,y_pred,pos_label=1)
    recall_s=recall_score(y_true,y_pred,pos_label=1)
    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(thresholds,recall[:-1],label='召回率',linewidth=0.5)
    plt.plot(thresholds, precision[:-1],label='准确率',linewidth=0.5)
    plt.plot([thresh,thresh],[0,1.2],linestyle='--',)
    plt.annotate(('precision=%0.4f' %pre_s),xy =(thresh,pre_s),xytext =(thresh,pre_s))
    plt.annotate(('recall=%0.4f' %recall_s),xy =(thresh,recall_s),xytext =(thresh, recall_s))
    plt.legend()
    plt.xlabel('切分阈值')
    plt.ylabel('比率')
    plt.ylim([0,1.05])
    plt.title('%s-模型不同阈值的准确率和召回率分布' %model_name)
    plt.savefig('%s-模型不同阈值的准确率和召回率分布.png' %model_name,dpi=600)
    plt.figure()
    plt.plot(precision,recall,linewidth=0.5,label=('F1-score=%0.4f' %f1_s))
    # plt.plot([pre_s, pre_s], [0, recall_s], linestyle='--', )
    # plt.plot([0, pre_s], [recall_s, recall_s], linestyle='--', )
    plt.xlabel('准确率')
    plt.ylabel('召回率')
    plt.legend()
    # plt.ylim([0, 1.05])
    # plt.xlim([0, 1.05])
    plt.title('%s-模型准确率和召回率联合分布' % model_name)
    plt.savefig('%s-模型准确率和召回率联合分布.png' % model_name, dpi=600)
    #ROC曲线和AUC
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    roc_auc=auc(fpr,tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=1, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳率（FPR）')
    plt.ylabel('真阳率（TPR）')
    plt.title('%s-模型ROC_AUC曲线' % model_name)
    plt.legend(loc="lower right")
    plt.savefig('%s-模型ROC_AUC曲线.png' % model_name, dpi=600)
    #confusion曲线
    def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        # plt.tight_layout()
        plt.ylabel('真实值')
        plt.xlabel('预测值')
        plt.savefig('%s.png' %title, dpi=600)
    cnf_matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cnf_matrix, classes=[0,1],title='%s-模型混淆矩阵' % model_name)
    plot_confusion_matrix(cnf_matrix, normalize=True,classes=[0, 1], title='%s-模型归一混淆矩阵' % model_name)
    # 改为原路径
    os.chdir(source_dir)

def tune_mlp(X_data,Y_data,starids):
    X_train, X_test, Y_train, Y_test, starids_train, starids_test = split_data(X_data, Y_data, starids, test_size=0.1)
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier()
    # 搜索参数
    param_dist = {
        'hidden_layer_sizes':[(3),(2)],
        'activation': ['logistic'],
        'solver':['adam'],
        'alpha': [0.00008,0.0001,0.0002],
        'learning_rate_init': [0.1],
        'batch_size': [256],
        'max_iter': [2000,5000]
    }
    from sklearn.model_selection import GridSearchCV
    search=GridSearchCV(clf,param_grid=param_dist,n_jobs=4,scoring='roc_auc')
    search.fit(X_train,Y_train)
    from sklearn.externals import joblib
    # 保存模型
    best_clf=search.best_estimator_
    best_clf.fit(X_train,Y_train)
    os.chdir(MODEL_PATH)
    joblib.dump(best_clf ,'best_mlp.model' )
    # 最佳超参数
    best_params=search.best_params_
    best_socre=search.best_score_
    print('-'*50)
    print('mlp最佳参数：')
    print(best_params)
    print('mlp最佳得分：')
    print(best_socre)
    print('-'*50)
    # 搜索结果
    # cv_result=pd.DataFrame(search.cv_results_)
    # cv_result.to_csv('mlp_cv_results.csv' ,index=False,sep='\t', encoding='gb18030')
    # # print('-' * 50)
    # # print('网格搜索结果')
    # # print(cv_result)
    # # 模型综合评估
    # y_pred_prob = search.predict_proba(X_test)
    # evaluate_model(Y_test, y_pred_prob, thresh=0.5, model_name='best_mlp')
    # # 模型分析
    # ana_model(X_test,Y_test,y_pred_prob,starids_test)
    return best_clf

# 调试gbdt 模型
def tune_gbdt(X_data,Y_data,starids):
    X_train, X_test, Y_train, Y_test,starids_train,starids_test=split_data(X_data,Y_data,starids,test_size=0.1)
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier()
    # 搜索参数
    param_dist={
        'loss':['deviance'],
        'n_estimators': [50,30,80],
        'max_depth': [3,2],
        'min_samples_split':[80,100],
        'min_samples_leaf': [20,30]
    }
    from sklearn.model_selection import GridSearchCV
    search=GridSearchCV(clf,param_grid=param_dist,n_jobs=4,scoring='roc_auc')
    search.fit(X_train,Y_train)
    from sklearn.externals import joblib
    # 保存模型
    best_clf=search.best_estimator_
    best_clf.fit(X_train, Y_train)
    os.chdir(MODEL_PATH)
    joblib.dump(best_clf ,'best_gbdt.model' )
    # 最佳超参数
    best_params=search.best_params_
    best_socre=search.best_score_
    print('-'*50)
    print('gbdt最佳参数：')
    print(best_params)
    print('gbdt最佳得分：')
    print(best_socre)
    print('-'*50)
    # 搜索结果
    # cv_result=pd.DataFrame(search.cv_results_)
    # cv_result.to_csv('gbdt_cv_results.csv' ,index=False,sep='\t', encoding='gb18030')
    # # print('-' * 50)
    # # print('网格搜索结果')
    # # print(cv_result)
    # # 模型综合评估
    # y_pred_prob = search.predict_proba(X_test)
    # evaluate_model(Y_test, y_pred_prob, thresh=0.5, model_name='best_GBDT')
    # # 模型分析
    # ana_model(X_test,Y_test,y_pred_prob,starids_test)
    return best_clf

# 训练mlp模型
def train_mlp(X_data,Y_data,starids):
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test, starids_train, starids_test = train_test_split(X_data, Y_data, starids, test_size=0.1)
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(activation = 'logistic',alpha = 0.0001,batch_size = 256,hidden_layer_sizes = 3
                        ,learning_rate_init = 0.1,max_iter = 2000,solver = 'adam')
    clf.fit(X_train,Y_train)
    from sklearn.externals import joblib
    # 保存模型
    os.chdir(MODEL_PATH)
    joblib.dump(clf ,'best_mlp.model' )
    return clf

# 训练gbdt 模型
def train_gbdt(X_data,Y_data,starids):
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test,starids_train,starids_test=train_test_split(X_data,Y_data,starids,test_size=0.1)
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(loss='deviance', max_depth=3, min_samples_leaf=30, min_samples_split=80,
                                     n_estimators=80)
    clf.fit(X_train,Y_train)
    from sklearn.externals import joblib
    # 保存模型
    os.chdir(MODEL_PATH)
    joblib.dump(clf ,'best_gbdt.model' )
    return clf

if __name__=='__main__':
    X_data,Y_data,starids=get_input()
    mlp = tune_mlp(X_data, Y_data, starids)
    gbdt = tune_gbdt(X_data, Y_data, starids)
    # mlp=train_mlp(X_data,Y_data,starids)
    # gbdt=train_gbdt(X_data, Y_data, starids)
    # logistic=tune_logistic(X_data, Y_data, starids)
pass
