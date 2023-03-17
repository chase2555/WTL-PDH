import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, confusion_matrix

data_df = pd.read_csv('dif_methods.csv')
y_test = data_df['ddg']
#train_feature= pd.read_csv("20221008_data.csv",usecols=['u-Non-polar-AB
test_pred = data_df['PremPDI']
# for i in range(len(test_pred)):
#     if test_pred[i] < 0.5:
#
#         test_pred[i] = 0
#     else:
#         test_pred[i] = 1

# test_pred = np.around(test_pred, 0).astype(int)
# train_pred = np.around(train_pred, 0).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
# test_pred = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0])
# tn=30
# fp=14
# tp=14
# fn=8
spe = tn / (tn + fp)
sen = tp / (tp + fn)
print(tn, fp, fn, tp)
print("spe:", spe)
print("sen", sen)
print("AUC:",roc_auc_score(y_test, test_pred))
print("test_ACC", accuracy_score(y_test, test_pred))
print('test_Precision', metrics.precision_score(y_test, test_pred))
print("test_MCC:", matthews_corrcoef(y_test, test_pred))
print('test_F1-score:', metrics.f1_score(y_test, test_pred))