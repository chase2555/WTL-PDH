import numpy as np
import pandas as pd
from imblearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold, StratifiedKFold
# from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, matthews_corrcoef, accuracy_score
from tensorflow.keras import layers, models
import json
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
#生成训练数据和标签
# train_data = np.random.rand(1000, 10)
# train_labels = np.random.randint(0, 2, size=(1000,))
# print(train_labels)
# print(train_data)
train_feature  = pd.read_csv('20221008_data.csv')
label_1= train_feature['ddg']
train_feature = train_feature.drop(['ddg','pdb','ref','pos','alt','chain'], axis=1)
#label_1= train_feature['ddg']
# 将数据集划分成10份
smo = SMOTE(random_state=114)
X, y = smo.fit_resample(train_feature, label_1)
# X=train_feature
# y=label_1
# from imblearn.over_sampling import RandomOverSampler
# over = RandomOverSampler(sampling_strategy=1)
# X, y = over.fit_resample(train_feature, label_1)
# from imblearn.over_sampling import ADASYN
# ada = ADASYN()
# X, y = ada.fit_resample(train_feature, label_1)

print(X)
print(y)
auc_scores = []
sen_scores = []
X = X.reset_index(drop=True).values
y = y.reset_index(drop=True).values
Spe = []
Sen= []
acc = []
pre = []
MCC = []
F1 = []
AUC1 = []

stratified_folder = StratifiedKFold(n_splits=10, shuffle=True, random_state=114)
for train_index, test_index in stratified_folder.split(X, y):
    X_train = X[train_index]
    Y_train = y[train_index]
    x_test = X[test_index]
    y_test = y[test_index]
    # 定义CNN模型
    model = models.Sequential([
        layers.Reshape((174, 1), input_shape=(174,)),
        layers.Conv1D(32, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # 编译模型
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    history = model.fit(X_train, Y_train, epochs=60, validation_data=(x_test, y_test))

    # 评估模型性能
    y_pred = model.predict(x_test)
    auc = roc_auc_score(y_test, y_pred)

    auc_scores.append(auc)
    test_pred=y_pred
    for i in range(len(test_pred)):
        if test_pred[i] < 0.5:

            test_pred[i] = 0
        else:
            test_pred[i] = 1
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spe = tn / (tn + fp)
    sen = tp / (tp + fn)
    Spe.append(spe)
    Sen.append(sen)
    acc.append(accuracy_score(y_test, test_pred))
    pre.append(metrics.precision_score(y_test, test_pred))
    F1.append(metrics.f1_score(y_test, test_pred))
    MCC.append(matthews_corrcoef(y_test, test_pred))



# 进行十次交叉

n=10
print("sen", np.sum(Sen) / n, "spe:", np.sum(Spe) / n, "pre:", np.sum(pre) / n, "F1:", np.sum(F1) / n,
      "MCC:", np.sum(MCC) / n, "ACC:", np.sum(acc) / n, "AUC:")
# 输出评估结果

print('AUC: %.3f%% (+/- %.3f%%)' % (np.mean(auc_scores)*100, np.std(auc_scores)*100))
print('Sen: %.3f%% (+/- %.3f%%)' % (np.mean(sen_scores)*100, np.std(sen_scores)*100))
