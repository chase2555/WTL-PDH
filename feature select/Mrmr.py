import pandas as pd
from sklearn.datasets import make_classification
data_df = pd.read_csv('')
y = data_df['ddg']
X = data_df.drop(['ddg','pdb','ref','pos','alt','chain'], axis=1)
X = pd.DataFrame(X)
y = pd.Series(y)
print (X)
print(y)
# select top 10 features using mRMR
from mrmr import mrmr_classif
selected_features = mrmr_classif(X=X, y=y, K=)
print(selected_features)