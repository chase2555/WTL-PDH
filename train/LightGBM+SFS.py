import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import pandas as pd
import numpy as np
import json
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
data_df = pd.read_csv('20221008_data.csv')
label_1 = data_df['ddg']
# train_feature = data_df.drop(['ddg','pdb','ref','pos','alt','chain'], axis=1)
# feature_name=['duTS13','DSSP23','ud17','DSSP15','duTS11','donor-num','pp-Total-Side-ABS','duTS19','DSSPPSI','d-p-s-ch-avg-CX','DSSP22','duTS20','IP_U','duTS8','all-num','duTS4','ud14','p-p-Total-Side-REL','d-Total-Side-ABS','d-p-Total-Side-REL','DSSP16','d-p-Total-Side-ABS','PSSM_R','DSSP24','d-All-atoms-ABS','base_ASA4','pp-All-atoms-ABS','d-p-average-CX','Nec','ud1','ud13','DSSP5','p-p-All-atoms-REL','ud11','d-All-polar-ABS','ud5','DSSP6','DSSP18','IP-ref','p-p-s-ch-avg-CX','ud6','base_ASA23','d-p-All-polar-REL','d-p-All-atoms-REL','Na','p-p-Total-Side-ABS','d-p-All-polar-ABS','d-p-All-atoms-ABS','d-Total-Side-REL','DSSPTCO','ud10','Mass','d-s-ch-avg-CX','ud3','Hdrpo','d-All-atoms-REL','DSSP21','duTS24','Ap21','DSSP10','p-p-All-atoms-ABS','Eccentricity','d-average-CX','p-p-Non-polar-REL','IP_C','Total-Side-REL','d-All-polar-REL','Ap11','ud12','Eiip']
# # print(len(feature_name))
#feature_name = ['base_ASA4','duTS13','DSSPTCO','d-s-ch-avg-CX','d-All-polar-ABS','DSSP18','DSSP23','d-p-All-polar-REL','Ap21','IP_U','ud6','DSSP22','ud17','DSSP15','duTS11','donor-num','pp-Total-Side-ABS','duTS19','DSSPPSI','d-p-s-ch-avg-CX','duTS20','duTS8','all-num','duTS4','ud14','p-p-Total-Side-REL','d-Total-Side-ABS','d-p-Total-Side-REL','DSSP16','d-p-Total-Side-ABS','PSSM_R','DSSP24','d-All-atoms-ABS','pp-All-atoms-ABS','d-p-average-CX','Nec','ud1','ud13','DSSP5','p-p-All-atoms-REL','ud11','ud5','DSSP6','IP-ref','p-p-s-ch-avg-CX','base_ASA23','d-p-All-atoms-REL','Na','p-p-Total-Side-ABS','d-p-All-polar-ABS','d-p-All-atoms-ABS','d-Total-Side-REL','ud10','Mass','ud3','Hdrpo','d-All-atoms-REL','DSSP21','duTS24','DSSP10','p-p-All-atoms-ABS','Eccentricity','d-average-CX','p-p-Non-polar-REL','IP_C','Total-Side-REL','d-All-polar-REL','Ap11','ud12','Eiip','ddg', 'pdb', 'ref', 'pos', 'alt', 'chain']
#20220929_feature_name = ['u_ASA_c_56_64_hz', 'ASA_shannon', 'd-Total-Side-ABS', 'ASA_node_2', 'u_ASA_node_5', 'ASA_b_log_energy', 'ASA_Ed', 'd-All-atoms-ABS', 'ASA_c_8_16_hz', 'u_ASA_c_48_56_hz', 'd-All-polar-ABS', 'u_ASA_b_log_energy', 'u_ASA_c_24_32_hz', 'd-Total-Side-REL', 'u_ASA_node_7', 'Total-Side-REL', 'ASA_sure', 'ASA_c_56_64_hz', 'ASA_coef_std', 'd-All-atoms-REL', 'u_ASA_Ea_2', 'ASA_b_threshold', 'd-Non-polar-ABS', 'ASA_node_5', 'u_ASA_b_threshold', 'u_ASA_node_3', 'u_ASA_sure', 'd-All-polar-REL', 'u_ASA_b_norm', 'u_ASA_c_32_40_hz', 'd-Non-polar-REL', 'u_ASA_Ea_3', 'ASA_norm', 'u_ASA_c_16_24_hz', 'ASA_b_shannon', 'u_ASA_sum_hz', 'u-All-polar-ABS', 'u_ASA_threshold', 'u-Total-Side-ABS', 'u_ASA_b_sure', 'u-All-atoms-ABS', 'u_ASA_node_4', 'All-polar-REL', 'ASA_b_norm', 'u_ASA_coef_std', 'All-atoms-REL', 'u_ASA_b_shannon', 'u_ASA_node_8', 'Total-Side-ABS', 'ASA_Ea_2', 'u_ASA_node_6', 'ASA_log_energy', 'ASA_c_16_24_hz', 'Non-polar-REL', 'u-Non-polar-ABS', 'ASA_node_8', 'ASA_sum_hz', 'ASA_b_sure', 'Non-polar-ABS', 'u_ASA_norm', 'u-All-atoms-REL', 'u_ASA_node_2', 'ASA_Ea_3', 'All-atoms-ABS', 'u_ASA_Ed', 'ASA_node_4', 'ASA_c_24_32_hz', 'u_ASA_c_40_48_hz', 'u-All-polar-REL', 'u-Non-polar-REL', 'u_ASA_shannon', 'u_ASA_c_8_16_hz', 'ASA_threshold', 'ASA_node_3', 'u_ASA_c_1_8_hz', 'u-Total-Side-REL', 'All-polar-ABS', 'ASA_c_40_48_hz', 'ASA_c_1_8_hz', 'u_ASA_coef_ave', 'u_ASA_node_1', 'u_ASA_Ea_1', 'ASA_c_32_40_hz', 'ASA_coef_ave', 'ASA_node_1', 'ASA_Ea_1', 'ASA_node_6', 'ASA_node_7', 'ASA_c_48_56_hz', 'u_ASA_log_energy','ddg', 'pdb', 'ref', 'pos', 'alt', 'chain']
#feature_name = [ 'u_ASA_c_8_16_hz', 'd_ASA_c_1_8_hz', 'u_ASA_c_56_64_hz', 'ASA_node_2', 'd_ASA_c_24_32_hz', 'd_ASA_node_5', 'ASA_Ed', 'd-Total-Side-ABS', 'ASA_b_log_energy', 'u_ASA_node_5', 'd_ASA_Ea_2', 'ASA_c_8_16_hz', 'd-All-atoms-ABS', 'd_ASA_node_1', 'd_ASA_Ea_3', 'ASA_coef_std', 'u_ASA_b_log_energy', 'd_ASA_node_3', 'd_ASA_Ea_1', 'd-All-polar-ABS', 'Total-Side-REL', 'd_ASA_coef_std', 'd_ASA_coef_ave', 'd_ASA_c_48_56_hz', 'ASA_b_threshold', 'd_ASA_c_16_24_hz', 'd-Total-Side-REL', 'u_ASA_node_7', 'u_ASA_c_24_32_hz', 'd_ASA_node_4', 'd-Non-polar-ABS', 'd_ASA_sure', 'u_ASA_Ea_2', 'ASA_shannon', 'd-All-atoms-REL', 'ASA_c_56_64_hz', 'd_ASA_node_6', 'u_ASA_b_threshold', 'u_ASA_c_48_56_hz', 'u_ASA_b_norm', 'ASA_node_5', 'u_ASA_c_32_40_hz', 'u_ASA_sure', 'd-Non-polar-REL', 'u_ASA_node_3', 'd-All-polar-REL', 'ASA_b_shannon', 'All-polar-REL', 'u_ASA_Ea_3', 'u_ASA_sum_hz', 'ASA_sure', 'u_ASA_threshold', 'u-All-polar-ABS', 'd_ASA_node_2', 'u_ASA_b_sure', 'u_ASA_c_16_24_hz', 'd_ASA_Ed', 'u-All-atoms-ABS', 'u-Total-Side-ABS', 'd_ASA_log_energy', 'd_ASA_shannon', 'u_ASA_node_4', 'All-atoms-REL', 'u_ASA_coef_std', 'ASA_b_norm', 'd_ASA_c_8_16_hz', 'u_ASA_node_8', 'u_ASA_b_shannon', 'd_ASA_c_40_48_hz', 'Total-Side-ABS', 'ASA_norm', 'ASA_Ea_2', 'd_ASA_sum_hz', 'd_ASA_b_sure', 'ASA_c_16_24_hz', 'u_ASA_node_6', 'd_ASA_b_norm', 'Non-polar-REL', 'u-Non-polar-ABS', 'd_ASA_b_threshold', 'ASA_node_8', 'ASA_log_energy', 'd_ASA_threshold', 'ASA_sum_hz', 'Non-polar-ABS', 'ASA_b_sure', 'd_ASA_norm', 'ASA_Ea_3', 'u_ASA_norm', 'u-All-atoms-REL', 'ASA_node_4', 'All-atoms-ABS', 'ASA_c_24_32_hz', 'u_ASA_node_2', 'u_ASA_Ed', 'u_ASA_c_40_48_hz', 'u-Non-polar-REL', 'u-All-polar-REL', 'ASA_node_3', 'u_ASA_shannon', 'd_ASA_c_32_40_hz', 'ASA_threshold', 'd_ASA_node_7', 'd_ASA_node_8', 'u-Total-Side-REL', 'u_ASA_c_1_8_hz', 'All-polar-ABS', 'd_ASA_b_shannon', 'ASA_c_40_48_hz', 'ASA_c_1_8_hz', 'u_ASA_coef_ave', 'u_ASA_node_1', 'ASA_c_32_40_hz', 'u_ASA_Ea_1', 'ASA_node_1', 'ASA_Ea_1', 'ASA_coef_ave', 'ASA_node_6', 'ASA_c_48_56_hz', 'd_ASA_b_log_energy', 'ASA_node_7', 'u_ASA_log_energy','ddg', 'pdb', 'ref', 'pos', 'alt', 'chain']
#feature_name=['dssp_b_shannon', 'ASA_coef_std', 'dssp_b_sure', 'u_ASA_c_56_64_hz', 'ASA_node_2', 'd_ASA_Ea_2', 'DSSPPSI', 'd-Total-Side-ABS', 'dssp_b_log_energy', 'ASA_Ed', 'd_ASA_node_5', 'd_ASA_c_1_8_hz', 'u_ASA_node_5', 'ASA_c_8_16_hz', 'd_ASA_c_24_32_hz', 'dssp_node_7', 'dssp_b_threshold', 'd-All-atoms-ABS', 'ASA_b_log_energy', 'd_ASA_Ea_3', 'd_ASA_node_1', 'dssp_node_8', 'u_ASA_b_log_energy', 'd_ASA_node_3', 'd_ASA_Ea_1', 'd-All-polar-ABS', 'Total-Side-REL', 'd_ASA_c_48_56_hz', 'dssp_Ea_1', 'd_ASA_coef_ave', 'dssp_b_norm', 'd_ASA_coef_std', 'd_ASA_c_16_24_hz', 'd-Total-Side-REL', 'ASA_b_threshold', 'u_ASA_node_7', 'd_ASA_node_4', 'u_ASA_c_24_32_hz', 'dssp_coef_ave', 'd-Non-polar-ABS', 'dssp_c_32_40_hz', 'ASA_shannon', 'u_ASA_Ea_2', 'd_ASA_sure', 'd-All-atoms-REL', 'd_ASA_node_6', 'dssp_c_40_48_hz', 'ASA_c_56_64_hz', 'u_ASA_c_48_56_hz', 'u_ASA_b_norm', 'dssp_Ed', 'DSSPTCO', 'ASA_node_5', 'dssp_sum_hz', 'u_ASA_b_threshold', 'dssp_node_2', 'u_ASA_sure', 'd-Non-polar-REL', 'u_ASA_node_3', 'dssp_node_1', 'dssp_c_8_16_hz', 'd-All-polar-REL', 'u_ASA_c_32_40_hz', 'u_ASA_Ea_3', 'ASA_sure', 'ASA_b_shannon', 'u_ASA_sum_hz', 'd_ASA_node_2', 'u_ASA_c_16_24_hz', 'u-All-polar-ABS', 'd_ASA_Ed', 'u_ASA_b_sure', 'dssp_threshold', 'u-Total-Side-ABS', 'u_ASA_threshold', 'u-All-atoms-ABS', 'd_ASA_shannon', 'd_ASA_log_energy', 'All-polar-REL', 'dssp_Ea_2', 'u_ASA_node_4', 'dssp_c_1_8_hz', 'DSSPACC', 'd_ASA_c_8_16_hz', 'dssp_c_56_64_hz', 'u_ASA_node_8', 'All-atoms-REL', 'u_ASA_coef_std', 'ASA_b_norm', 'u_ASA_b_shannon', 'd_ASA_c_40_48_hz', 'dssp_node_5', 'dssp_node_6', 'ASA_norm', 'Total-Side-ABS', 'dssp_c_48_56_hz', 'd_ASA_sum_hz', 'ASA_Ea_2', 'd_ASA_b_sure', 'u_ASA_node_6', 'ASA_c_16_24_hz', 'd_ASA_b_norm', 'Non-polar-REL', 'd_ASA_b_threshold', 'u-Non-polar-ABS', 'ASA_node_8', 'DSSPPHI', 'ASA_log_energy', 'd_ASA_norm', 'd_ASA_threshold', 'dssp_sure', 'ASA_sum_hz', 'dssp_c_16_24_hz', 'ASA_b_sure', 'dssp_c_24_32_hz', 'Non-polar-ABS', 'DSSPKAPPA', 'ASA_Ea_3', 'u_ASA_norm', 'u-All-atoms-REL', 'ASA_node_4', 'All-atoms-ABS', 'u_ASA_node_2', 'u_ASA_Ed', 'ASA_c_24_32_hz', 'u_ASA_c_40_48_hz', 'u-Non-polar-REL', 'u-All-polar-REL', 'u_ASA_shannon', 'd_ASA_c_32_40_hz', 'dssp_coef_std', 'u_ASA_c_8_16_hz', 'ASA_threshold', 'ASA_node_3', 'd_ASA_node_8', 'd_ASA_node_7', 'u-Total-Side-REL', 'u_ASA_c_1_8_hz', 'dssp_Ea_3', 'All-polar-ABS', 'd_ASA_b_shannon', 'ASA_c_40_48_hz', 'ASA_c_1_8_hz', 'DSSPALPHA', 'dssp_node_3', 'dssp_node_4', 'u_ASA_coef_ave', 'u_ASA_node_1', 'u_ASA_Ea_1', 'ASA_c_32_40_hz', 'dssp_shannon', 'dssp_log_energy', 'ASA_node_1', 'ASA_Ea_1', 'ASA_coef_ave', 'ASA_node_6', 'ASA_c_48_56_hz', 'd_ASA_b_log_energy', 'ASA_node_7', 'dssp_norm', 'u_ASA_log_energy','ddg', 'pdb', 'ref', 'pos', 'alt', 'chain']
#feature_name=['dssp_b_shannon', 'ASA_coef_std', 'donor-num', 'DSSPPSI', 'd-s-ch-avg-DPX', 'u_ASA_c_56_64_hz', 'dssp_b_log_energy', 'ASA_node_2', 'd_ASA_Ea_2', 'd-Total-Side-ABS', 'd_ASA_node_5', 'ASA_Ed', 'd_ASA_c_1_8_hz', 'u_ASA_node_5', 'ASA_c_8_16_hz', 'dssp_b_threshold', 'd_ASA_c_24_32_hz', 'dssp_node_7', 'd-All-atoms-ABS', 'ASA_b_log_energy', 'd_ASA_Ea_3', 'd_ASA_node_1', 'dssp_b_norm', 'dssp_node_8', 'u_ASA_b_log_energy', 'd_ASA_node_3', 'd_ASA_Ea_1', 'd-All-polar-ABS', 'dssp_Ea_1', 'd_ASA_c_48_56_hz', 'd_ASA_coef_ave', 'd_ASA_coef_std', 'Total-Side-REL', 'd_ASA_c_16_24_hz', 'd-Total-Side-REL', 'ASA_b_threshold', 'd_ASA_node_4', 'u_ASA_node_7', 'd-s-ch-avg-CX', 'dssp_c_32_40_hz', 'dssp_coef_ave', 'u_ASA_c_24_32_hz', 'ASA_shannon', 'd_ASA_sure', 'u_ASA_Ea_2', 'd-All-atoms-REL', 'dssp_c_40_48_hz', 'd_ASA_node_6', 'd-Non-polar-ABS', 'ASA_c_56_64_hz', 'dssp_Ed', 'u_ASA_b_threshold', 'dssp_sum_hz', 'u_ASA_c_48_56_hz', 'DSSPTCO', 'd-average-CX', 'u_ASA_b_norm', 'ASA_node_5', 'u_ASA_sure', 'dssp_node_2', 'u_ASA_node_3', 'dssp_node_1', 'd-Non-polar-REL', 'dssp_c_8_16_hz', 'd-average-DPX', 'u_ASA_c_32_40_hz', 'd-All-polar-REL', 'dssp_b_sure', 'u_ASA_Ea_3', 'u_ASA_sum_hz', 'd_ASA_node_2', 'ASA_b_shannon', 'u_ASA_c_16_24_hz', 'd_ASA_Ed', 'u_ASA_b_sure', 'ASA_sure', 'u-All-polar-ABS', 'u-s-ch-avg-CX', 'dssp_threshold', 'u_ASA_threshold', 'd_ASA_log_energy', 'u-Total-Side-ABS', 'd_ASA_shannon', 'u-All-atoms-ABS', 'dssp_Ea_2', 'u-s-ch-avg-DPX', 'All-polar-REL', 'u_ASA_node_4', 'dssp_c_1_8_hz', 'u-average-CX', 'd_ASA_c_8_16_hz', 'dssp_c_56_64_hz', 'DSSPACC', 'u_ASA_node_8', 'ASA_b_norm', 'u_ASA_coef_std', 'u-average-DPX', 'dssp_node_5', 'u_ASA_b_shannon', 'd_ASA_c_40_48_hz', 'All-atoms-REL', 'dssp_node_6', 'd_ASA_sum_hz', 'dssp_c_48_56_hz', 'ASA_Ea_2', 'd_ASA_b_sure', 'ASA_norm', 'Total-Side-ABS', 'u_ASA_node_6', 'd_ASA_b_norm', 'ASA_c_16_24_hz', 's-ch-avg-DPX', 's-ch-avg-CX', 'd_ASA_b_threshold', 'Non-polar-REL', 'ASA_node_8', 'DSSPPHI', 'ASA_log_energy', 'u-Non-polar-ABS', 'd_ASA_norm', 'd_ASA_threshold', 'ASA_sum_hz', 'dssp_sure', 'ASA_b_sure', 'dssp_c_16_24_hz', 'dssp_c_24_32_hz', 'DSSPKAPPA', 'u_ASA_norm', 'Non-polar-ABS', 'ASA_Ea_3', 'ASA_node_4', 'u-All-atoms-REL', 'u_ASA_node_2', 'u_ASA_Ed', 'All-atoms-ABS', 'ASA_c_24_32_hz', 'u_ASA_c_40_48_hz', 'u_ASA_shannon', 'u-All-polar-REL', 'u_ASA_c_8_16_hz', 'u-Non-polar-REL', 'dssp_coef_std', 'd_ASA_c_32_40_hz', 'ASA_node_3', 'd_ASA_node_8', 'ASA_threshold', 'd_ASA_node_7', 'u_ASA_c_1_8_hz', 'u-Total-Side-REL', 'dssp_Ea_3', 'average-CX', 'All-polar-ABS', 'd_ASA_b_shannon', 'ASA_c_40_48_hz', 'ASA_c_1_8_hz', 'DSSPALPHA', 'dssp_node_3', 'dssp_node_4', 'u_ASA_coef_ave', 'u_ASA_node_1', 'u_ASA_Ea_1', 'dssp_shannon', 'ASA_c_32_40_hz', 'dssp_log_energy', 'ASA_node_1', 'ASA_Ea_1', 'ASA_coef_ave', 'average-DPX', 'ASA_node_6', 'ASA_c_48_56_hz', 'd_ASA_b_log_energy', 'ASA_node_7', 'dssp_norm', 'u_ASA_log_energy','ddg', 'pdb', 'ref', 'pos', 'alt', 'chain']
rate = [0.1]
ran=[]
Au=[]
for rat in rate:
    for g in range(1):
        #feature_name = ['dssp_b_shannon', 'ASA_coef_std', 'donor-num', 'DSSPPSI', 'd-s-ch-avg-DPX', 'u_ASA_c_56_64_hz',
                        # 'dssp_b_log_energy', 'ASA_node_2', 'd_ASA_Ea_2', 'd-Total-Side-ABS', 'd_ASA_node_5', 'ASA_Ed',
                        # 'd_ASA_c_1_8_hz', 'u_ASA_node_5', 'ASA_c_8_16_hz', 'dssp_b_threshold', 'd_ASA_c_24_32_hz',
                        # 'dssp_node_7', 'd-All-atoms-ABS', 'ASA_b_log_energy', 'd_ASA_Ea_3', 'd_ASA_node_1',
                        # 'dssp_b_norm', 'dssp_node_8', 'u_ASA_b_log_energy', 'd_ASA_node_3', 'd_ASA_Ea_1',
                        # 'd-All-polar-ABS', 'dssp_Ea_1', 'd_ASA_c_48_56_hz', 'd_ASA_coef_ave', 'd_ASA_coef_std',
                        # 'Total-Side-REL', 'd_ASA_c_16_24_hz', 'd-Total-Side-REL', 'ASA_b_threshold', 'd_ASA_node_4',
                        # 'u_ASA_node_7', 'd-s-ch-avg-CX', 'dssp_c_32_40_hz', 'dssp_coef_ave', 'u_ASA_c_24_32_hz',
                        # 'ASA_shannon', 'd_ASA_sure', 'u_ASA_Ea_2', 'd-All-atoms-REL', 'dssp_c_40_48_hz', 'd_ASA_node_6',
                        # 'd-Non-polar-ABS', 'ASA_c_56_64_hz', 'dssp_Ed', 'u_ASA_b_threshold', 'dssp_sum_hz',
                        # 'u_ASA_c_48_56_hz', 'DSSPTCO', 'd-average-CX', 'u_ASA_b_norm', 'ASA_node_5', 'u_ASA_sure',
                        # 'dssp_node_2', 'u_ASA_node_3', 'dssp_node_1', 'd-Non-polar-REL', 'dssp_c_8_16_hz',
                        # 'd-average-DPX', 'u_ASA_c_32_40_hz', 'd-All-polar-REL', 'dssp_b_sure', 'u_ASA_Ea_3',
                        # 'u_ASA_sum_hz', 'd_ASA_node_2', 'ASA_b_shannon', 'u_ASA_c_16_24_hz', 'd_ASA_Ed', 'u_ASA_b_sure',
                        # 'ASA_sure', 'u-All-polar-ABS', 'u-s-ch-avg-CX', 'dssp_threshold', 'u_ASA_threshold',
                        # 'd_ASA_log_energy', 'u-Total-Side-ABS', 'd_ASA_shannon', 'u-All-atoms-ABS', 'dssp_Ea_2',
                        # 'u-s-ch-avg-DPX', 'All-polar-REL', 'u_ASA_node_4', 'dssp_c_1_8_hz', 'u-average-CX',
                        # 'd_ASA_c_8_16_hz', 'dssp_c_56_64_hz', 'DSSPACC', 'u_ASA_node_8', 'ASA_b_norm', 'u_ASA_coef_std',
                        # 'u-average-DPX', 'dssp_node_5', 'u_ASA_b_shannon', 'd_ASA_c_40_48_hz', 'All-atoms-REL',
                        # 'dssp_node_6', 'd_ASA_sum_hz', 'dssp_c_48_56_hz', 'ASA_Ea_2', 'd_ASA_b_sure', 'ASA_norm',
                        # 'Total-Side-ABS', 'u_ASA_node_6', 'd_ASA_b_norm', 'ASA_c_16_24_hz', 's-ch-avg-DPX',
                        # 's-ch-avg-CX', 'd_ASA_b_threshold', 'Non-polar-REL', 'ASA_node_8', 'DSSPPHI', 'ASA_log_energy',
                        # 'u-Non-polar-ABS', 'd_ASA_norm', 'd_ASA_threshold', 'ASA_sum_hz', 'dssp_sure', 'ASA_b_sure',
                        # 'dssp_c_16_24_hz', 'dssp_c_24_32_hz', 'DSSPKAPPA', 'u_ASA_norm', 'Non-polar-ABS', 'ASA_Ea_3',
                        # 'ASA_node_4', 'u-All-atoms-REL', 'u_ASA_node_2', 'u_ASA_Ed', 'All-atoms-ABS', 'ASA_c_24_32_hz',
                        # 'u_ASA_c_40_48_hz', 'u_ASA_shannon', 'u-All-polar-REL', 'u_ASA_c_8_16_hz', 'u-Non-polar-REL',
                        # 'dssp_coef_std', 'd_ASA_c_32_40_hz', 'ASA_node_3', 'd_ASA_node_8', 'ASA_threshold',
                        # 'd_ASA_node_7', 'u_ASA_c_1_8_hz', 'u-Total-Side-REL', 'dssp_Ea_3', 'average-CX',
                        # 'All-polar-ABS', 'd_ASA_b_shannon', 'ASA_c_40_48_hz', 'ASA_c_1_8_hz', 'DSSPALPHA',
                        # 'dssp_node_3', 'dssp_node_4', 'u_ASA_coef_ave', 'u_ASA_node_1', 'u_ASA_Ea_1', 'dssp_shannon',
                        # 'ASA_c_32_40_hz', 'dssp_log_energy', 'ASA_node_1', 'ASA_Ea_1', 'ASA_coef_ave', 'average-DPX',
                        # 'ASA_node_6', 'ASA_c_48_56_hz', 'd_ASA_b_log_energy', 'ASA_node_7', 'dssp_norm',
                        # 'u_ASA_log_energy', 'ddg', 'pdb', 'ref', 'pos', 'alt', 'chain']
        feature_name =['dssp_b_shannon', 'DSSPPSI', 'd_ASA_b_sure', 'dssp_b_log_energy', 'ASA_node_2', 'ASA_Ed', 'u_ASA_c_56_64_hz',
         'u_ASA_node_5', 'd-s-ch-avg-CX', 'd_ASA_sum_hz', 'd_ASA_c_1_8_hz', 'u-All-polar-ABS', 'd_ASA_b_log_energy',
         'd_ASA_Ed', 'DSSPALPHA', 'd-All-polar-ABS', 'd-average-CX', 'Total-Side-ABS', 'ASA_c_8_16_hz', 'd_ASA_node_5',
         'dssp_c_32_40_hz', 'd_ASA_b_norm', 'd_ASA_Ea_3', 'u_ASA_b_sure', 'dssp_node_5', 'ASA_c_48_56_hz', 'ASA_sum_hz',
         'dssp_node_7', 'dssp_Ea_3', 'dssp_node_2', 'dssp_log_energy', 'd_ASA_node_7', 'd-Total-Side-REL',
         'dssp_b_sure', 'd_ASA_node_1', 'u-All-polar-REL', 'ASA_Ea_3', 'd_ASA_node_3', 'u_ASA_b_shannon',
         'u_ASA_sum_hz', 'd_ASA_c_32_40_hz', 'u_ASA_node_1', 'u-s-ch-avg-CX', 'ASA_node_8', 'dssp_coef_ave',
         'd_ASA_shannon', 'd_ASA_coef_ave', 'u-average-CX', 'ASA_b_log_energy', 'ASA_sure', 'dssp_b_norm',
         'dssp_node_8', 'DSSPKAPPA', 'ASA_c_1_8_hz', 'd_ASA_c_8_16_hz', 'u_ASA_sure', 'd_ASA_log_energy',
         'u_ASA_b_norm', 'd_ASA_coef_std', 'DSSPPHI', 'u_ASA_node_8', 'u_ASA_c_1_8_hz', 'u_ASA_log_energy',
         'u_ASA_Ea_1', 's-ch-avg-DPX', 'd_ASA_c_48_56_hz', 'dssp_c_8_16_hz', 'u_ASA_b_log_energy', 'Non-polar-REL',
         'average-CX', 'dssp_node_1', 'ASA_Ea_2', 'u_ASA_Ed', 'd_ASA_Ea_2', 'DSSPTCO', 'u_ASA_Ea_3', 'All-atoms-ABS',
         'dssp_c_16_24_hz', 'ASA_coef_std', 'd_ASA_node_8', 'u_ASA_c_32_40_hz', 'u_ASA_node_7', 'd_ASA_norm',
         'dssp_c_24_32_hz', 'ASA_shannon', 'd-Total-Side-ABS', 'dssp_c_40_48_hz', 'dssp_node_3', 'ASA_c_40_48_hz',
         'dssp_sum_hz', 'd_ASA_Ea_1', 'd-Non-polar-REL', 'ASA_norm', 's-ch-avg-CX', 'dssp_Ed', 'All-atoms-REL',
         'u_ASA_c_40_48_hz', 'ASA_c_16_24_hz', 'Total-Side-REL', 'u-Total-Side-REL', 'u-average-DPX', 'd_ASA_node_6',
         'dssp_Ea_2', 'u_ASA_Ea_2', 'd-All-polar-REL', 'All-polar-REL', 'dssp_coef_std', 'dssp_node_4', 'ASA_node_4',
         'd_ASA_b_shannon', 'average-DPX', 'ASA_node_1', 'd_ASA_sure', 'ASA_c_56_64_hz', 'd_ASA_node_2',
         'dssp_c_48_56_hz', 'ASA_log_energy', 'u-Total-Side-ABS', 'dssp_Ea_1', 'd_ASA_c_24_32_hz', 'ASA_coef_ave',
         'All-polar-ABS', 'd-All-atoms-REL', 'DSSPACC', 'ASA_c_24_32_hz', 'u_ASA_norm', 'u-Non-polar-REL',
         'dssp_c_56_64_hz', 'u_ASA_coef_ave',
         'dssp_c_1_8_hz', 'u_ASA_node_2', 'Non-polar-ABS', 'd-All-atoms-ABS', 'u_ASA_c_48_56_hz',
         'ASA_node_5', 'ASA_node_6', 'ASA_node_7', 'd-average-DPX', 'dssp_node_6', 'ASA_b_norm', 'd_ASA_node_4',
         'ASA_b_sure', 'dssp_shannon', 'u_ASA_shannon', 'd-s-ch-avg-DPX', 'ASA_c_32_40_hz', 'dssp_sure',
         'd_ASA_c_16_24_hz', 'ASA_node_3',
         'ASA_b_shannon', 'd_ASA_c_40_48_hz', 'd-Non-polar-ABS', 'u_ASA_node_4', 'u_ASA_c_24_32_hz', 'u_ASA_node_6',
         'dssp_norm', 'u_ASA_c_16_24_hz', 'u_ASA_c_8_16_hz', 'u-All-atoms-ABS', 'ASA_Ea_1', 'u_ASA_node_3',
         'u_ASA_coef_std', 'u-Non-polar-ABS', 'u-All-atoms-REL', 'dssp_threshold', 'donor-num', 'dssp_b_threshold',
         'd_ASA_b_threshold', 'u-s-ch-avg-DPX', 'ASA_b_threshold', 'ASA_threshold', 'u_ASA_b_threshold',
         'u_ASA_threshold', 'd_ASA_threshold','ddg', 'pdb', 'ref', 'pos', 'alt', 'chain']

        i = 0
        A = []
        a = 0
        A.append(a)
        a = 1
        E = []
        B = []

        while (i < 20):
            print(i)
            b = i
            sen1 = []
            spe1 = []
            pre1 = []
            F11 = []
            MCC1 = []
            ACC1 = []
            AUC1 = []
            if feature_name[i - 1] == 'ddg':
                break
            while (i > 0):
                if (A[a - 1] < max(A)):
                    dele = feature_name[i - 1]
                    feature_name.remove(dele)
                    print(dele)
                    feature_name.append(dele)
                    i = i
                    break
                else:
                    E.append(feature_name[i - 1])
                    i = i + 1

                    break
            a = a + 1

            if i == 0:
                i = i + 1

            print(i)
            train_feature = data_df.drop(feature_name[i:len(feature_name)], axis=1)
            # i=25
            # i = i + 1
            if i == 20:
                i = i + 88
            print(train_feature)
            print("max_AUC:", max(A))
            from imblearn.over_sampling import SMOTE

            smo = SMOTE(random_state=73)
            # smo = SMOTE()
            X, y = smo.fit_resample(train_feature, label_1)
            # from imblearn.over_sampling import RandomOverSampler
            # over = RandomOverSampler(sampling_strategy=1)
            # X, y = over.fit_resample(train_feature, label_1)
            # from imblearn.over_sampling import ADASYN
            # ada = ADASYN()
            # X, y = ada.fit_resample(train_feature, label_1)

            #X=train_feature
            #y=label_1
            print(X)
            print(y)
            # print("random:", i)
            X = X.reset_index(drop=True).values
            y = y.reset_index(drop=True).values

            for s in range(10):
                acc = []
                Spe = []
                Sen = []
                pre = []
                rec = []
                F1 = []
                MCC = []
                AUC = []
                n = 10

                stratified_folder = StratifiedKFold(n_splits=10, shuffle=False)
                for train_index, test_index in stratified_folder.split(X, y):

                    X_train = X[train_index]
                    Y_train = y[train_index]
                    x_test = X[test_index]
                    y_test = y[test_index]


                    def train_model(X_train, Y_train, x_test, y_test, Sen, Spe, acc, pre, rec, F1, MCC, AUC, rat):
                        # 创建成lgb特征的数据集格式,将使加载更快
                        lgb_train = lgb.Dataset(X_train, label=Y_train)
                        lgb_eval = lgb.Dataset(x_test, label=y_test, reference=lgb_train)
                        parameters = {
                            'task': 'train',
                            'max_depth': 15,
                            'boosting_type': 'gbdt',
                            'num_leaves': 50,  # 叶子节点数
                            'n_estimators': 1000,
                            'objective': 'binary',
                            'metric': 'auc',
                            'learning_rate': 0.1,
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
                                              verbose_eval=10,
                                              # feature_name=features_name
                                              )

                        test_pred = gbm_model.predict(x_test, num_iteration=gbm_model.best_iteration)
                        train_pred = gbm_model.predict(X_train, num_iteration=gbm_model.best_iteration)

                        test_roc_auc_score = roc_auc_score(y_test, test_pred)
                        train_roc_auc_score = roc_auc_score(Y_train, train_pred)  # Y_train
                        print(test_pred)

                        for q in range(len(test_pred)):
                            if test_pred[q] < 0.5:

                                test_pred[q] = 0
                            else:
                                test_pred[q] = 1

                        # test_pred = np.around(test_pred, 0).astype(int)
                        # train_pred = np.around(train_pred, 0).astype(int)

                        tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
                        # print(tn, fp, fn, tp)
                        if accuracy_score(y_test, test_pred) > 0 or test_roc_auc_score > 0:
                            # print(y_test)
                            # print(test_pred)
                            spe = tn / (tn + fp)
                            sen = tp / (tp + fn)
                            Spe.append(spe)
                            Sen.append(sen)
                            # print(tn, fp, fn, tp)
                            # print("spe:", spe)
                            # print("sen", sen)
                            AUC.append(test_roc_auc_score)
                            # print("test_ACC", accuracy_score(y_test, test_pred))

                            acc.append(accuracy_score(y_test, test_pred))
                            # print('test_Precision', metrics.precision_score(y_test, test_pred))
                            pre.append(metrics.precision_score(y_test, test_pred))
                            # print('test_Recall', metrics.recall_score(y_test, test_pred))
                            # rec.append(metrics.recall_score(y_test, test_pred))
                            # print('test_F1-score:', metrics.f1_score(y_test, test_pred))
                            F1.append(metrics.f1_score(y_test, test_pred))
                            # print("test_MCC:", matthews_corrcoef(y_test, test_pred))
                            # print("test_AUC:", test_roc_auc_score)
                            MCC.append(matthews_corrcoef(y_test, test_pred))

                            # print("train_ACC", accuracy_score(Y_train, train_pred))
                            # print('train_Precision', metrics.precision_score(Y_train, train_pred))
                            # print('train_Recall', metrics.recall_score(Y_train, train_pred))
                            # print('train_F1-score:', metrics.f1_score(Y_train, train_pred))
                            # print("train_MCC:", matthews_corrcoef(Y_train,train_pred))
                            # print("train_AUC:", train_roc_auc_score)
                            # gbm_model.save_model('model.txt')
                            # gbm_model.save_model('smote_340_mrmr_70_model' + str(test_roc_auc_score*100)+ '.txt')
                            # gbm_model.save_model('zz-fold' + str(test_roc_auc_score * 100) + '.txt')
                            # print(Spe, Sen, acc, pre, F1, MCC, AUC)
                            # plt.figure(figsize=(12, 6))
                            # lgb.plot_importance(gbm_model, max_num_features=30)
                            # plt.title("Featurertances")
                            # plt.show()

                        return gbm_model, evals_result


                    model, evals_result = train_model(X_train, Y_train, x_test, y_test, Sen, Spe, acc, pre, rec, F1,
                                                      MCC,
                                                      AUC, rat)
                sen1.append(np.sum(Sen) / n)
                spe1.append(np.sum(Spe) / n)
                pre1.append(np.sum(pre) / n)
                F11.append(np.sum(F1) / n)
                MCC1.append(np.sum(MCC) / n)
                ACC1.append(np.sum(acc) / n)
                AUC1.append(np.sum(AUC) / n)

            # if np.sum(AUC1) / n >0:
            print("sen1", np.sum(sen1) / n, "spe1:", np.sum(Spe) / n, "pre1:", np.sum(pre) / n, "F11:", np.sum(F1) / n,
                  "MCC1:", np.sum(MCC) / n, "ACC1:", np.sum(acc) / n, "AUC1:", np.sum(AUC) / n)
            # a = rat

            y = np.sum(AUC1) / n
            # print(y)
            A.append(y)
            print("max_AUC:", max(A))
            c = b
            # print(rat)
            # B.append(b)

            print(A)



        print(A)
        print(B)
        del (A[-1])
        print(max(A))
        ran.append(g)
        Au.append(max(A))
        print(a)
        print(b)
        print(max(Au))
        print(ran[Au.index(max(Au))])
        print(Au)
        print(ran)
        print(train_feature)
        print(E)
        import os
        with open('test_2.txt', 'a') as file0:
            print("Au:"+str(max(A))+"ran"+str(g), file=file0)
            print(str(E)+"\n", file=file0)

print(max(Au))
print(ran[Au.index(max(Au))])
print(Au)
print(ran)
print(train_feature)
print(X)
print(E)
print(Au)

