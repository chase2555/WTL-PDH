import lightgbm as lgb
import pandas as pd
import numpy as np
import json
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
######
train_data = pd.read_csv('20221008_data.csv')
#X = train_data.drop(['ddg','pdb','ref','pos','chain','alt'], axis=1)
X = pd.read_csv("20221008_data.csv",usecols=['u-Non-polar-ABS', 'd-Non-polar-ABS','ASA_node_2','u_ASA_c_32_40_hz','u_ASA_c_56_64_hz', 'u_ASA_sum_hz', 'u_ASA_Ea_3', 'd_ASA_node_8','d_ASA_Ea_2', 'DSSPPSI', 'dssp_b_shannon', 'dssp_b_log_energy', 'dssp_sure',  'd-average-DPX', 'd-s-ch-avg-CX'])
#X = train_data.drop(['d-Total-Side-ABS', 'd_ASA_node_5', 'ASA_Ed', 'u_ASA_node_5', 'ASA_c_8_16_hz', 'd_ASA_c_24_32_hz', 'dssp_node_7', 'd-All-atoms-ABS', 'ASA_b_log_energy', 'd_ASA_Ea_3', 'dssp_b_norm', 'u_ASA_b_log_energy', 'd_ASA_node_3', 'd_ASA_Ea_1', 'd-All-polar-ABS', 'dssp_Ea_1', 'd_ASA_c_48_56_hz', 'd_ASA_coef_ave', 'd_ASA_coef_std', 'Total-Side-REL', 'd_ASA_c_16_24_hz', 'd-Total-Side-REL', 'ASA_b_threshold', 'd_ASA_node_4', 'u_ASA_node_7', 'dssp_c_32_40_hz', 'dssp_coef_ave', 'u_ASA_c_24_32_hz', 'ASA_shannon', 'd_ASA_sure', 'u_ASA_Ea_2', 'd-All-atoms-REL', 'dssp_c_40_48_hz', 'd_ASA_node_6', 'd-Non-polar-ABS', 'ASA_c_56_64_hz', 'dssp_Ed', 'u_ASA_b_threshold', 'dssp_sum_hz', 'u_ASA_c_48_56_hz', 'DSSPTCO', 'd-average-CX', 'u_ASA_b_norm', 'ASA_node_5', 'u_ASA_sure', 'dssp_node_2', 'u_ASA_node_3', 'dssp_node_1', 'd-Non-polar-REL', 'dssp_c_8_16_hz', 'd-average-DPX', 'u_ASA_c_32_40_hz', 'd-All-polar-REL', 'dssp_b_sure', 'u_ASA_Ea_3', 'u_ASA_sum_hz', 'd_ASA_node_2', 'ASA_b_shannon', 'u_ASA_c_16_24_hz', 'd_ASA_Ed', 'u_ASA_b_sure', 'ASA_sure', 'u-All-polar-ABS', 'u-s-ch-avg-CX', 'dssp_threshold', 'u_ASA_threshold', 'd_ASA_log_energy', 'u-Total-Side-ABS', 'd_ASA_shannon', 'u-All-atoms-ABS', 'dssp_Ea_2', 'u-s-ch-avg-DPX', 'All-polar-REL', 'u_ASA_node_4', 'dssp_c_1_8_hz', 'u-average-CX', 'd_ASA_c_8_16_hz', 'dssp_c_56_64_hz', 'DSSPACC', 'u_ASA_node_8', 'ASA_b_norm', 'u_ASA_coef_std', 'u-average-DPX', 'dssp_node_5', 'u_ASA_b_shannon', 'd_ASA_c_40_48_hz', 'All-atoms-REL', 'dssp_node_6', 'd_ASA_sum_hz', 'dssp_c_48_56_hz', 'ASA_Ea_2', 'd_ASA_b_sure', 'ASA_norm', 'Total-Side-ABS', 'u_ASA_node_6', 'd_ASA_b_norm', 'ASA_c_16_24_hz', 's-ch-avg-DPX', 's-ch-avg-CX', 'd_ASA_b_threshold', 'Non-polar-REL', 'ASA_node_8', 'DSSPPHI', 'ASA_log_energy', 'u-Non-polar-ABS', 'd_ASA_norm', 'd_ASA_threshold', 'ASA_sum_hz', 'dssp_sure', 'ASA_b_sure', 'dssp_c_16_24_hz', 'dssp_c_24_32_hz', 'DSSPKAPPA', 'u_ASA_norm', 'Non-polar-ABS', 'ASA_Ea_3', 'ASA_node_4', 'u-All-atoms-REL', 'u_ASA_node_2', 'u_ASA_Ed', 'All-atoms-ABS', 'ASA_c_24_32_hz', 'u_ASA_c_40_48_hz', 'u_ASA_shannon', 'u-All-polar-REL', 'u_ASA_c_8_16_hz', 'u-Non-polar-REL', 'dssp_coef_std', 'd_ASA_c_32_40_hz', 'ASA_node_3', 'd_ASA_node_8', 'ASA_threshold', 'd_ASA_node_7', 'u_ASA_c_1_8_hz', 'u-Total-Side-REL', 'dssp_Ea_3', 'average-CX', 'All-polar-ABS', 'd_ASA_b_shannon', 'ASA_c_40_48_hz', 'ASA_c_1_8_hz', 'DSSPALPHA', 'dssp_node_3', 'dssp_node_4', 'u_ASA_coef_ave', 'u_ASA_node_1', 'u_ASA_Ea_1', 'dssp_shannon', 'ASA_c_32_40_hz', 'dssp_log_energy', 'ASA_node_1', 'ASA_Ea_1', 'ASA_coef_ave', 'average-DPX', 'ASA_node_6', 'ASA_c_48_56_hz', 'd_ASA_b_log_energy', 'ASA_node_7', 'dssp_norm', 'u_ASA_log_energy','ddg', 'pdb', 'ref', 'pos', 'alt', 'chain'],axis=1)
#X = train_data.drop(['dssp_b_sure', 'DSSPPSI', 'd-Total-Side-ABS', 'dssp_b_log_energy', 'ASA_Ed', 'd_ASA_node_5', 'u_ASA_node_5', 'ASA_c_8_16_hz', 'd_ASA_c_24_32_hz', 'dssp_node_7', 'dssp_b_threshold', 'd-All-atoms-ABS', 'ASA_b_log_energy', 'd_ASA_Ea_3', 'd_ASA_node_1', 'dssp_node_8', 'u_ASA_b_log_energy', 'd_ASA_node_3', 'd_ASA_Ea_1', 'd-All-polar-ABS', 'Total-Side-REL', 'd_ASA_c_48_56_hz', 'dssp_Ea_1', 'd_ASA_coef_ave', 'dssp_b_norm', 'd_ASA_coef_std', 'd_ASA_c_16_24_hz', 'd-Total-Side-REL', 'ASA_b_threshold', 'u_ASA_node_7', 'd_ASA_node_4', 'u_ASA_c_24_32_hz', 'dssp_coef_ave', 'd-Non-polar-ABS', 'dssp_c_32_40_hz', 'ASA_shannon', 'u_ASA_Ea_2', 'd_ASA_sure', 'd-All-atoms-REL', 'd_ASA_node_6', 'dssp_c_40_48_hz', 'ASA_c_56_64_hz','u_ASA_b_norm', 'dssp_Ed', 'ASA_node_5', 'dssp_node_2', 'u_ASA_sure', 'd-Non-polar-REL', 'u_ASA_node_3', 'dssp_node_1', 'dssp_c_8_16_hz', 'd-All-polar-REL', 'u_ASA_c_32_40_hz', 'u_ASA_Ea_3', 'ASA_sure', 'ASA_b_shannon', 'u_ASA_sum_hz', 'd_ASA_node_2', 'u_ASA_c_16_24_hz', 'u-All-polar-ABS', 'd_ASA_Ed', 'u_ASA_b_sure', 'dssp_threshold', 'u-Total-Side-ABS', 'u_ASA_threshold', 'u-All-atoms-ABS', 'd_ASA_shannon', 'd_ASA_log_energy', 'All-polar-REL', 'dssp_Ea_2', 'u_ASA_node_4', 'dssp_c_1_8_hz', 'DSSPACC', 'd_ASA_c_8_16_hz', 'dssp_c_56_64_hz', 'u_ASA_node_8', 'All-atoms-REL', 'u_ASA_coef_std', 'ASA_b_norm', 'u_ASA_b_shannon', 'd_ASA_c_40_48_hz', 'dssp_node_5', 'dssp_node_6', 'ASA_norm', 'Total-Side-ABS', 'dssp_c_48_56_hz', 'd_ASA_sum_hz', 'ASA_Ea_2', 'd_ASA_b_sure', 'u_ASA_node_6', 'ASA_c_16_24_hz', 'd_ASA_b_norm', 'Non-polar-REL', 'd_ASA_b_threshold', 'u-Non-polar-ABS', 'ASA_node_8', 'DSSPPHI', 'ASA_log_energy', 'd_ASA_norm', 'd_ASA_threshold', 'dssp_sure', 'ASA_sum_hz', 'dssp_c_16_24_hz', 'ASA_b_sure', 'dssp_c_24_32_hz', 'Non-polar-ABS', 'DSSPKAPPA', 'ASA_Ea_3', 'u_ASA_norm', 'u-All-atoms-REL', 'ASA_node_4', 'All-atoms-ABS', 'u_ASA_node_2', 'u_ASA_Ed', 'ASA_c_24_32_hz', 'u_ASA_c_40_48_hz', 'u-Non-polar-REL', 'u-All-polar-REL', 'u_ASA_shannon', 'd_ASA_c_32_40_hz', 'dssp_coef_std', 'u_ASA_c_8_16_hz', 'ASA_threshold', 'ASA_node_3', 'd_ASA_node_8', 'd_ASA_node_7', 'u-Total-Side-REL', 'u_ASA_c_1_8_hz', 'dssp_Ea_3', 'All-polar-ABS', 'd_ASA_b_shannon', 'ASA_c_40_48_hz', 'ASA_c_1_8_hz', 'DSSPALPHA', 'dssp_node_3', 'dssp_node_4', 'u_ASA_coef_ave', 'u_ASA_node_1', 'u_ASA_Ea_1', 'ASA_c_32_40_hz', 'dssp_shannon', 'dssp_log_energy', 'ASA_node_1', 'ASA_Ea_1', 'ASA_coef_ave', 'ASA_node_6', 'ASA_c_48_56_hz', 'd_ASA_b_log_energy', 'ASA_node_7', 'dssp_norm', 'u_ASA_log_energy','ddg', 'pdb', 'ref', 'pos', 'alt', 'chain'], axis=1)
y = train_data['ddg']
print(X)
print(y)
from imblearn.over_sampling import SMOTE
# 定义SMOTE模型，random_state相当于随机数种子的作用
#random_state=6
Au=[]
for i in range(1):
    smo = SMOTE(random_state=114)
    ran=i-1
    print(ran)
    X_train, y_train = smo.fit_resample(X, y)
    print(X_train)
    print(y_train)
    # print(X_train)
    # print(y_train)

    test_data = pd.read_csv('20221008_test_data.csv')
    y_test = test_data['ddg']
    #X_test = test_data.drop(['ddg','pdb','ref','pos','chain','alt'], axis=1)
    X_test = pd.read_csv("20221008_test_data.csv", usecols=['u-Non-polar-ABS', 'd-Non-polar-ABS','ASA_node_2','u_ASA_c_32_40_hz','u_ASA_c_56_64_hz', 'u_ASA_sum_hz', 'u_ASA_Ea_3', 'd_ASA_node_8','d_ASA_Ea_2', 'DSSPPSI', 'dssp_b_shannon', 'dssp_b_log_energy', 'dssp_sure',  'd-average-DPX', 'd-s-ch-avg-CX'])


    print(X_test)
    print(y_test)
    def train_model(X_train, Y_train, x_test, y_test, ac, pre, rec, F1, MCC, AUC):
        # 创建成lgb特征的数据集格式,将使加载更快
        lgb_train = lgb.Dataset(X_train, label=Y_train)
        print(X_train)
        print(Y_train)
        lgb_eval = lgb.Dataset(x_test, label=y_test, reference=lgb_train)
        #print(x_test)
        #print(y_test)
        parameters = {
            'task': 'train',
            'max_depth': 15,
            'boosting_type': 'gbdt',
            'num_leaves': 50,  # 叶子节点数
            'n_estimators': 1000,
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.32,
            'feature_fraction': 0.7,  # 小于 1.0, LightGBM 将会在每次迭代中随机选择部分特征.
            'bagging_fraction': 1,  # 类似于 feature_fraction, 但是它将在不进行重采样的情况下随机选择部分数据
            'bagging_freq': 15,  # bagging 的频率, 0 意味着禁用 bagging. k 意味着每 k 次迭代执行bagging
            'lambda_l1': 0.5,
            'lambda_l2': 0,
            'cat_smooth': 10,  # 用于分类特征,这可以降低噪声在分类特征中的影响, 尤其是对数据很少的类别
            'is_unbalance': False,  # 适合二分类。这里如果设置为True，评估结果降低3个点
            'verbose': 0
        }

        evals_result = {}  # 记录训练结果所用
        gbm_model = lgb.train(parameters,
                              lgb_train,
                              valid_sets=[lgb_train, lgb_eval],
                              num_boost_round=1000,  # 提升迭代的次数
                              early_stopping_rounds=5,
                              evals_result=evals_result,
                              verbose_eval=10
                              )

        test_pred = gbm_model.predict(x_test, num_iteration=gbm_model.best_iteration)
        train_pred = gbm_model.predict(X_train, num_iteration=gbm_model.best_iteration)
       # print(test_pred)

        test_roc_auc_score = roc_auc_score(y_test, test_pred)
        train_roc_auc_score = roc_auc_score(Y_train, train_pred)  # Y_train
        print(test_pred)

        for i in range(len(test_pred)):
            if test_pred[i] < 0.5:

                test_pred[i] = 0
            else:
                test_pred[i] = 1

        # test_pred = np.around(test_pred, 0).astype(int)
        # train_pred = np.around(train_pred, 0).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
        print(tn, fp, fn, tp)
        print(y_test)
        print(test_pred)
        if accuracy_score(y_test, test_pred) > 0.0 or test_roc_auc_score > 0.50:

            spe = tn / (tn + fp)
            sen = tp / (tp + fn)
            print(tn, fp, fn, tp)
            print("spe:", spe)
            print("sen", sen)
            AUC.append(test_roc_auc_score)
            print("test_ACC", accuracy_score(y_test, test_pred))

            ac.append(accuracy_score(y_test, test_pred))
            print('test_Precision', metrics.precision_score(y_test, test_pred))
            pre.append(metrics.precision_score(y_test, test_pred))
            # print('test_Recall', metrics.recall_score(y_test, test_pred))
            # rec.append(metrics.recall_score(y_test, test_pred))
            print('test_F1-score:', metrics.f1_score(y_test, test_pred))
            F1.append(metrics.f1_score(y_test, test_pred))
            print("test_MCC:", matthews_corrcoef(y_test, test_pred))
            print("test_AUC:", test_roc_auc_score)
            MCC.append(matthews_corrcoef(y_test, test_pred))
            Au.append(test_roc_auc_score)

            # print("train_ACC", accuracy_score(Y_train, train_pred))
            # print('train_Precision', metrics.precision_score(Y_train, train_pred))
            # print('train_Recall', metrics.recall_score(Y_train, train_pred))
            # print('train_F1-score:', metrics.f1_score(Y_train, train_pred))
            # print("train_MCC:", matthews_corrcoef(Y_train,train_pred))
            # print("train_AUC:", train_roc_auc_score)
            # gbm_model.save_model('model.txt')
            gbm_model.save_model(str(test_roc_auc_score * 100) + '_XXX.txt')
            # gbm_model.save_model('1.txt')
            # print(ac, pre, rec, F1, MCC, AUC)
        return gbm_model, evals_result


    ac = []
    pre = []
    rec = []
    F1 = []
    MCC = []
    AUC = []
    for i in range(1):
        model, evals_result = train_model(X_train, y_train, X_test, y_test, ac, pre, rec, F1, MCC, AUC)

print(max(Au))