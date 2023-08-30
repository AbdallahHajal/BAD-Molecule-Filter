# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 15:37:06 2023

@author: abdallah.abouhajal
"""

# Built-in and third-party libraries
import warnings
import base64
import io
from distutils.command.upload import upload
from scipy import stats
from PIL import Image
import pickle

# Numpy, Pandas, Matplotlib, Seaborn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import image
import matplotlib.patches as mpatches

# Sklearn modules
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score, confusion_matrix,
                             r2_score, mean_squared_error, roc_curve, roc_auc_score)

# Imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# RDKIt modules
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

# Streamlit
import streamlit as st

# Other third-party modules
from mordred import Calculator, descriptors
import dask.dataframe as dd

# ML frameworks
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


import streamlit as st
from PIL import Image

def home_page():
    # Introduction about Colloidal Aggregates
    st.markdown('**Colloidal Aggregates** are a serious problem in drug discovery and drug development projects. Aggregators bind to various enzymes non-specifically (Fig. 1) and produce wrong results during assay screening, making them a source of false positives. As a result, they may lead to the loss of years of work, effort, and resources. This work aims to build a novel webapp that can distinguish between colloidal aggregators and non-aggregators in chemical space. This Web App is developed by training the three largest datasets available using LightGBM and CatBoost algorithms. Predictions from the three datasets are combined into a consensus model.')
    
    st.markdown('---')  # Horizontal Line
    
    figure1 = Image.open('Figure 1..JPG')
    st.image(figure1, caption='Figure 1. Process of drug aggregation & enzyme-aggregate complex ', width=800)

    st.markdown('---')  # Horizontal Line

    # Displaying the Graphical Abstract
    figure2 = Image.open('figure 2.JPG')
    st.image(figure2, caption='Figure 2. Graphical Abstract', width=900)
    
    st.markdown('---')  # Horizontal Line

    # Instructions on how to use the app
    st.subheader('How to Use this App')
    st.markdown('To predict the aggregation behavior of your molecules, please select "**Prediction Tool**" from the dropdown menu in the **sidebar**. This method employs the Universal Strategy, which is based on the majority voting mechanism of three individual models, each constructed from large datasets. For a deeper understanding and additional details, please refer to the accompanying research article.')


with open('model_trial_CRUUUZAAAIN.pkl','rb') as f:
        model_Cruz = pickle.load(f)
with open('model_trial_Laccccccc.pkl','rb') as f:
        model_Lac = pickle.load(f)
with open('Newest_Model_Shoichet&ZINC.pkl','rb') as f:
        model_Shoichet = pickle.load(f)        
with open('scaler_Cruzain.pkl','rb') as f:
        scaler_Cru = pickle.load(f)
with open('scaler_Lactamase.pkl','rb') as f:
        scaler_Lac = pickle.load(f)
with open('scaler_Shoichet.pkl','rb') as f:
        scaler_Shoi = pickle.load(f)





descriptor_columns_cru=["SMR", "GhoseFilter", "FilterItLogS","Col_1239", "piPC2", "Col_197", "n6aHRing",
"SLogP", "Col_980", "ATS2m", "Mi", "SMR_VSA1", "Col_1774", "NaasC","Col_491",
"GATS1i", "Col_1970", "nBase", "Col_488", "Col_1866", "Col_1775", "Col_516",
"NssNH", "Col_724", "NaasN", "WPath", "SlogP_VSA2", "Col_385", "Col_1865", "Col_702", 
"GGI7", "Col_55", "SddsN", "SMR_VSA5", "MPC10", "Col_920", "Col_80", "StN", "Xc-5d", 
"Col_653", "AATS0dv", "ZMIC3", "EState_VSA9", "Col_879", "ECIndex", "Col_1852", "NaaaC",
"GGI9", "VSA_EState6","Col_97", "AATS1dv", "Col_1791", "ATSC4v", "Col_1983", "Col_995",
"Col_591", "GATS1dv", "nHBDon", "SdsCH", "Col_798", "Col_255", "GATS1are", "GATS1p", 
"Col_33", "AATS1i", "Col_845", "ATS1dv", "Col_598", "Col_232", "Col_2019", "Col_405", 
"Col_1217", "SaaN", "Col_1959", "Col_1957", "Col_694", "TSRW10", "Col_1677", 
"EState_VSA10", "Col_1357", "Col_1867", "Col_1443","PEOE_VSA12",
"SdsN", "nAHRing", "Col_1136", "Xch-5dv", "Col_381", "ATS6d", "Col_1964", "Col_349",
"Col_145", "SlogP_VSA8", "Col_1275", "Col_1693","ATSC2p", "VSA_EState9", "Col_813", 
"SaaNH", "ATSC1i", "ATSC1v", "SlogP_VSA1","Col_110", "Col_787", "Col_852", "ATS3are", 
"Col_1858", "Col_2003", "MIC1","Col_1602", "TopoPSA", "Col_362", "Col_1724", "Col_641",
"Col_1160", "Col_695","Col_1342","Col_1266", "Col_216",
"VSA_EState4", "Col_147", "Col_1804", "Col_374", "Col_486", "GGI10", 
"nBondsKD", "BalabanJ", "SaaS", "Col_1444", "EState_VSA3", "Col_1405", "PEOE_VSA4", 
"nAcid", "Col_1934", "Col_456", "Col_1304", "Col_1024", "SaasC", "EState_VSA8", 
"FCSP3", "Col_203","Col_314", "ATSC1dv", "SssO", "piPC8", "Col_837", "Col_222", 
"GGI5", "Col_407", "SlogP_VSA3", "Col_1713", "JGI4", "RotRatio", "Col_881", "JGI1", 
"Col_1739", "nS","Col_1948", "Col_202", "Col_675", "EState_VSA4", "SaaO", "SlogP_VSA10",
"AATSC1p", "ATSC3i", "Col_267", "Col_2044", "PEOE_VSA2", "Col_983", "ATSC0dv", 
"EState_VSA1", "SMR_VSA3", "ATSC1d", "VSA_EState3", "MPC3", "SsNH2", "Col_1601", "Col_1326", "Col_842", "SssCH2", "nBondsD", "Col_1676", "Xch-6dv", "Col_670",
"Col_1424", "Col_1982", "Xc-5dv","Col_1707", "Col_586", "Col_1200", 
"Col_896", "n5AHRing", "Col_1119", "C1SP3", "Col_1855", "Col_966", "Col_1398", "ATS0m",
"PEOE_VSA1", "Col_950", "Col_651", "Col_223", "SMR_VSA6", "ZMIC1", "Col_1951", 
"Diameter", "Col_162", "Col_186", "ATSC2d", "nO", "Col_295", "Col_814", "Col_1504", 
"nG12FHRing", "Col_285", "Col_31", "Col_213", "Col_464", "Col_1738", "Col_982", 
"naHRing", "Col_689", "MATS1dv", "Col_316", "Col_1969", "Col_1665", "n6Ring", 
"Col_1449", "Col_1469", "SRW07", "nRing", "Col_875", "Col_838", "Col_1651", "SssS", 
"Col_1816", "Col_1499", "PEOE_VSA11", "n5HRing", "Col_1407", "Col_1898", "Col_1427", 
"ATSC0pe", "Col_34", "nHBAcc", "TopoPSA(NO)", "GATS1Z", "Mp", "Col_1771", "Col_409", 
"ATSC4m", "Col_233", "Col_430", "Col_1827", "Col_1770", "PEOE_VSA13", "Col_1497", 
"nARing", "Col_94", "Col_355", "JGI3", "nAromAtom", "Xc-3d", "SMR_VSA7", "SMR_VSA9", 
"SlogP_VSA4", "AATS1are", "Col_1564", "Col_705", "Col_231", "VSA_EState1", "Col_866",
"Col_843", "Col_1709", "Col_342", "Col_286", "Col_906", "Col_962", "Xc-6d", "Col_338",
"Col_1989", "GGI6", "nG12FaHRing", "Col_860", "Col_725", "Col_504", "n12FaHRing", 
"Col_692", "Col_638", "Col_45", "Col_41", "Col_974", "Col_732", "Col_490", "Col_42", 
"nFAHRing", "Col_496", "Col_608", "Col_59", "Col_774", "Col_926", "Col_416", "Col_25",
"Col_463", "Col_644", "NdssC", "Col_991", "ATSC2m", "n5ARing", "Col_831",
"Col_1997", "Col_656", "ATS7m", "Col_210", "Col_277", "JGI5", "Col_268", 
"ATSC0d", "Col_1905", "nFRing", "Col_593", "Xc-3dv", "MIC5", "Xch-6d", "JGI6", 
"SMR_VSA4", "Col_278", "Col_623", "SdO", "Col_561", "Col_781", "Col_747", "Col_56", 
"Col_579", "Col_479", "GGI4", "Col_588", "Col_856", "AATS1m", "ATSC2pe", "Col_474", 
"Col_921", "nX", "Col_514", "VSA_EState2", "Col_1195", "Col_853", "Col_799", "nHRing",
"Col_549", "Col_590", "n11FAHRing", "GGI2", "Col_92", "Xch-5d", 
"n10FaRing", "Col_949", "NsOH", "JGI7", "Col_999", "NdNH", "SsssCH", "nRot", "Col_913",
"Xpc-6dv", "VSA_EState7", "ATS8dv", "Col_929"]


descriptor_columns_Lac=['MWC01', 'nHBDon', 'SLogP', 'ATS0p', 'SlogP_VSA2', 'FilterItLogS', 'nBondsD', 'VSA_EState8', 'Col_2009', 'fragCpx','GGI9', 'ATSC1i', 'piPC9', 'SdO', 'Col_1602', 'nX', 'SMR_VSA3', 'PEOE_VSA8', 'ATSC3i', 'TMPC10', 'FCSP3', 'MATS1v', 'ATSC0i', 'ATS5p', 'AATSC1pe', 'ATSC2v', 'EState_VSA9', 'ATSC2i', 'ATS7m', 'nAcid', 'piPC1', 'Col_216', 'TopoPSA', 'NdS', 'Col_1849', 'MIC3', 'SlogP_VSA3', 'SlogP_VSA7','VSA_EState3', 'WPath', 'VSA_EState2', 'Col_798', 'ATS0m','ATSC3dv', 'Col_738','Col_1549', 'PEOE_VSA6', 'nBase', 'Col_1417', 'SMR_VSA6', 'Col_714', 'PEOE_VSA9', 'SaaO', 'Col_1089', 'Col_1160', 'IC2', 'ATSC1Z', 'SsssN', 'AATS1i', 'TSRW10', 'Col_222', 'SsCl', 'SlogP_VSA8', 'C2SP2', 'Col_148', 'BalabanJ', 'AATSC0p', 'IC5', 'ATSC6i', 'Col_1604', 'SlogP_VSA1', 'Col_277', 'Col_74', 'Col_1816', 'Col_314', 'ATSC5i', 'AATS1v', 'Xch-7d', 'NaaN', 'ATSC8i', 'GATS1i', 'PEOE_VSA3', 'Col_378','Col_473', 'ATSC3Z', 'PEOE_VSA13', 'VSA_EState5', 'GATS1m',  'Col_401', 'SsssNH', 'ATSC4dv', 'Xp-3dv', 'ATSC2dv','Col_50', 'Col_495', 'Xch-6dv', 'Xc-5d', 'AATS0p', 'nBondsA', 'Col_1804', 'ATSC4Z', 'TopoPSA(NO)', 'VSA_EState9', 'Col_273', 'Col_1825', 'SdssC', 'GATS1v', 'ATSC4d', 'JGI6', 'Col_1603', 'Col_1653', 'Col_949', 'SdsssP', 'ATSC1dv', 'Col_695', 'PEOE_VSA11', 'SMR_VSA9', 'ATSC7p','ATS3Z', 'IC0', 'Col_1250', 'MATS1dv', 'ZMIC1', 'ATSC5d', 'Col_942',  'Col_639', 'Col_831', 'Col_1508', 'Col_12', 'Col_82', 'ATSC3d', 'ATSC5Z', 'Col_423', 'JGI5', 'Col_2038', 'Col_532', 'Col_656', 'JGI4', 'Col_809', 'PEOE_VSA4', 'Col_145', 'PEOE_VSA1', 'SaasN', 'Col_1979', 'Col_586', 'Col_155', 'Col_1389', 'Col_167', 'ATSC6d', 'C3SP2', 'ATSC4v', 'Col_522', 'Col_1041', 'ATSC8are', 'ATSC1d', 'Col_52', 'Col_1217', 'PEOE_VSA12', 'JGI2', 'Col_1918', 'fMF', 'VSA_EState7', 'Xch-7dv',  'SMR_VSA1', 'Col_521', 'Col_41', 'Xpc-5d', 'PEOE_VSA10', 'Col_1780','SssssC', 'Col_1504', 'Col_1130', 'ATSC5v', 'ATSC7v', 'n5aRing', 'ATSC6dv', 'Col_405', 'Col_1934',  'Col_42', 'Col_980', 'Col_1416', 'ATSC8p', 'Col_396', 'piPC3', 'Col_239', 'ATSC5p', 'Col_845', 'ATSC8d', 'Col_1135', 'Col_679', 'Col_1409', 'Col_251', 'Col_233', 'Col_39', 'nSpiro', 'Col_1861', 'AATSC0dv', 'Col_602', 'JGI8', 'MATS1p', 'Col_1014', 'ATSC4are', 'Col_974', 'ATSC3v', 'GATS1dv', 'AATSC1d', 'ATSC2d', 'PEOE_VSA2', 'Col_268', 'CIC4', 'MATS1are', 'ATSC0p', 'ECIndex', 'Col_490', 'Col_783', 'Col_1430', 'ATSC7i', 'Col_1097', 'Col_92', 'Col_1177', 'Col_1761', 'ZMIC3', 'Col_1622', 'Col_1342', 'ATS8dv', 'Col_433', 'Col_1488', 'Col_106', 'ATSC7dv', 'Col_2020', 'Col_623', 'ATSC7Z',  'Col_278', 'Col_515', 'Col_1329', 'C1SP2', 'Col_1245', 'Col_1624', 'Col_636', 'GGI1', 'Col_597', 'EState_VSA8', 'Col_247', 'SlogP_VSA10', 'Col_1145', 'SMR_VSA5', 'Col_576', 'Col_504', 'ATSC0pe', 'Col_94', 'Col_162', 'NssO', 'Col_789', 'GATS1p', 'IC1', 'Col_13', 'SsBr', 'Col_1022', 'Xc-3d', 'PEOE_VSA7', 'Col_1889', 'SlogP_VSA11', 'Col_420', 'ATSC8m', 'Col_1065', 'SaasC', 'Col_1310', 'ATSC5dv', 'Col_1084', 'RotRatio', 'Col_1964', 'SIC0', 'Col_140', 'Col_1660', 'NdsN', 'Col_1845', 'Col_417', 'Col_1453', 'Col_1866', 'Col_245', 'Col_227', 'Col_1012', 'Col_506', 'Col_1047', 'Col_1063', 'Col_923', 'Col_1944', 'Col_17', 'Col_645', 'Col_1142', 'Col_207', 'Col_1806',  'ATSC6p', 'Col_7', 'JGI10', 'Col_885', 'Col_611', 'AATS1Z','Col_1517', 'Col_369', 'Col_1341', 'Col_451', 'Col_397', 'Diameter', 'Xpc-6dv', 'Col_2041', 'Col_950', 'Col_1260', 'AATSC0Z', 'Col_771', 'Col_1385', 'Col_113', 'SlogP_VSA4', 'Col_1254', 'ATSC8dv', 'ATSC6Z', 'Col_26', 'Col_1353', 'Col_1411', 'Col_781', 'Col_779', 'Col_1444', 'VSA_EState4', 'Col_867', 'CIC1', 'Col_30', 'AXp-1dv', 'ATSC4p', 'Col_334', 'EState_VSA10', 'Col_1835', 'Col_1356', 'MWC04', 'nRot', 'Col_1634', 'Col_1449', 'Col_255', 'Col_457', 'Col_487', 'Col_965', 'Col_2006', 'Col_2013', 'Col_951', 'Col_1850', 'Mi', 'Col_1966', 'Col_1765', 'Col_1395', 'Col_1006', 'ATSC7d', 'Col_339', 'Col_1921', 'nHetero', 'Col_1357', 'ATSC6v', 'Col_1847', 'Xch-5d', 'Col_1750', 'ATS2dv', 'Col_561', 'Col_1295', 'Col_102', 'Col_667', 'Col_80', 'ATS6dv', 'ATSC6pe', 'Col_1476', 'Col_863', 'Col_1911', 'Col_901', 'ATSC4i', 'Col_896', 'Col_893', 'AATSC0v', 'Col_1425', 'SssS', 'Col_1920', 'nFARing', 'Col_1611', 'Col_1694', 'NaasC', 'Col_620', 'AATSC0i', 'Col_1281', 'MIC0', 'Col_832', 'Col_1363', 'Col_1261', 'Col_1775', 'Col_1317', 'Xc-4dv', 'n9FaRing', 'Col_1490', 'JGI7', 'Col_1617', 'Col_1121', 'ATSC3p', 'Col_1060', 'SlogP_VSA6', 'Col_1656', 'Col_488', 'EState_VSA2', 'n9FRing', 'Col_670', 'Col_1695', 'Col_808', 'Col_1481', 'AXp-1d','Col_1312', 'Col_1701', 'Col_613', 'Col_784', 'Xch-6d', 'Col_1612', 'n5Ring', 'Col_1627', 'Col_1055', 'Col_1163', 'Col_1897', 'Col_1445', 'C4SP3', 'Col_2044', 'nHRing', 'VSA_EState6', 'Col_1296', 'Col_1736', 'Col_826', 'Col_1802', 'Col_1589', 'SaaS', 'Col_1017', 'Col_1152', 'Col_1513', 'Col_961', 'SaaaC', 'Col_1675', 'JGI3', 'Col_496', 'Col_1309', 'Col_844', 'Col_823', 'Col_1930', 'Col_1967', 'SlogP_VSA5', 'Col_1469', 'GGI3', 'SMR_VSA4', 'Col_618', 'Col_1716', 'Col_1745', 'JGI9', 'NaasN', 'Col_1600', 'Col_1005', 'Xc-3dv', 'Col_1387', 'Col_1077', 'Col_1322', 'n6aHRing', 'ATSC0d', 'Col_1212', 'Col_637', 'Col_1238', 'Col_1522', 'Xc-4d', 'Col_849', 'Col_1833', 'Col_1514', 'ZMIC2', 'Col_540', 'Col_1667', 'Col_1932', 'TopoShapeIndex', 'Col_1304', 'Col_2028', 'Col_1637', 'Col_1087', 'Col_1057', 'Col_1034', 'Col_1638', 'Col_1905', 'Col_1749', 'Col_1298', 'Col_1482', 'Col_1757', 'Col_1741', 'Col_1655', 'Col_1848', 'Col_1792', 'Col_2029', 'Col_1235', 'Col_1114', 'n6HRing', 'Col_1105', 'Col_1981', 'Col_1463', 'Col_1115', 'AATSC0d', 'Col_1362', 'Xc-5dv', 'Col_1722', 'Col_1032', 'Col_1129', 'PEOE_VSA5', 'Col_1713', 'Col_1904', 'Col_1498', 'MZ', 'Col_1516', 'Col_1033', 'Col_1810', 'Col_1950', 'CIC2', 'Mpe', 'Col_1754', 'Col_1914', 'Col_1050', 'Col_1519', 'Col_1443', 'n6ARing', 'n4aRing', 'Col_1291', 'ATSC8v', 'Col_1240', 'SMR_VSA2', 'Col_1000', 'Col_1586', 'Col_996', 'n5AHRing', 'Col_1592', 'Col_1308', 'Col_1194', 'Col_1441', 'Col_1860', 'Col_1193', 'Col_1676', 'Col_1405', 'EState_VSA4', 'AATSC0are', 'Col_1066', 'Col_1480', 'Col_1956', 'NsssNH', 'Col_1791', 'nG12FAHRing', 'Col_1783', 'Col_1998','Xch-5dv', 'GhoseFilter', 'Col_1705', 'Col_2025', 'Col_2017', 'Col_1620', 'Col_2039', 'n6Ring', 'Col_1855', 'Col_1630', 'Col_1719', 'Col_1668', 'SRW09', 'StsC', 'Col_1503', 'Col_1601', 'nN', 'nARing', 'Col_1925', 'Xc-6dv', 'Col_1560', 'Col_1768', 'Col_1731', 'Col_1984', 'ATSC2m', 'Col_1585', 'Col_1865', 'Col_1610', 'n5ARing', 'Col_1827', 'Col_1727', 'Col_1698', 'Xc-6d', 'Col_1641', 'C3SP3', 'Col_1779', 'nO', 'Col_674', 'Col_1970', 'Col_1959', 'Col_1971', 'n12FAHRing', 'n6aRing', 'Col_2042', 'EState_VSA7', 'Col_1975', 'nS', 'Xch-4d', 'Col_1982', 'Col_2001']


descriptor_columns_Shoi=['SLogP', 'FilterItLogS', 'SddsN', 'MATS1v', 'AATS1p', 'SaaN', 'nBondsD', 'MATS1i', 'SaaO', 'C2SP2', 'Col_694', 'Diameter', 'Col_1867', 'NdsN', 'SMR_VSA3', 'SlogP_VSA7', 'SlogP_VSA1', 'PEOE_VSA4', 'GATS1p', 'Col_110', 'Lipinski', 'GATS1v', 'AATS1i', 'SlogP_VSA2', 'EState_VSA8', 'nAcid', 'BalabanJ', 'ATSC3are', 'VSA_EState3', 'GATS1i', 'GATS1are', 'GGI10', 'VSA_EState6', 'JGI1', 'ATSC5dv', 'Mp', 'PEOE_VSA9', 'Xch-6d', 'n4aRing', 'PEOE_VSA7', 'piPC10', 'ZMIC1', 'AATSC0Z', 'MIC5', 'AATS0i', 'AATSC1are', 'SdO', 'SMR_VSA7', 'Col_31', 'ATSC4are', 'NdS', 'GATS1dv', 'AATSC1Z', 'Col_1449', 'ATSC2m', 'MATS1Z', 'SlogP_VSA3', 'SaaaC', 'VSA_EState2', 'AATSC0d', 'n5aHRing', 'Col_656', 'JGI6', 'SdssC', 'nHRing', 'SMR_VSA2', 'ATSC2d', 'Xch-6dv', 'ATSC8are', 'GGI5', 'AATS1dv', 'PEOE_VSA10', 'ATS0m', 'GGI3', 'ATSC5i', 'EState_VSA2', 'ATSC7v', 'SlogP_VSA10', 'SMR_VSA4', 'ATS3dv', 'MW', 'ECIndex', 'ATSC6d', 'ATSC3dv', 'JGI5', 'SRW09', 'PEOE_VSA11', 'ATSC4d', 'Col_1745', 'JGI2', 'AATSC0v', 'Col_1195', 'Xp-2dv', 'ATSC4p', 'ZMIC4', 'Col_162', 'VSA_EState5', 'SssssC', 'Xp-5d', 'Col_1866', 'ATSC4i', 'NsssNH', 'Xpc-4dv', 'ATSC2i', 'SsssN', 'RotRatio', 'PEOE_VSA8', 'AATSC0dv', 'CIC4', 'SlogP_VSA8', 'AATSC1dv', 'ATS6v', 'SMR_VSA1', 'Xc-4dv', 'Xpc-4d', 'Xc-5dv', 'Col_1749', 'ATS8dv', 'Col_1038', 'ATSC7pe', 'VSA_EState7', 'ATSC3p', 'ATSC3Z', 'PEOE_VSA1', 'ATS0p', 'Xc-3d', 'ATSC6pe', 'AATS0p', 'SaasN', 'ATSC8i', 'EState_VSA9', 'fMF', 'MPC10', 'Col_1210', 'ATSC6m', 'Col_1823', 'CIC0', 'AATSC0i', 'Col_750', 'PEOE_VSA6', 'Col_859', 'ATSC8m', 'GATS1m', 'Xc-3dv', 'Col_45', 'SMR_VSA6', 'ZMIC2', 'ATSC6v', 'SsssNH', 'ATSC5pe', 'Col_1331', 'AATSC1d', 'JGI10', 'ATSC4dv', 'SaasC', 'Xch-5dv', 'Col_1947', 'JGI7', 'ATSC8d', 'ATSC6i', 'TopoPSA', 'ATSC6dv', 'NaaS', 'ATSC6p', 'Xch-5d', 'ATSC5m', 'JGI4', 'Col_1847', 'ATSC2dv', 'Col_1174', 'Col_167', 'SlogP_VSA5', 'AATS0d', 'Xc-5d', 'ATSC2p', 'n6Ring', 'TopoPSA(NO)', 'ATSC2are', 'JGI3', 'Col_1472', 'Col_1416', 'Col_1316', 'Col_617', 'Col_264', 'ATSC5d', 'Col_753', 'SaaS', 'ATSC3i', 'ATSC3v', 'VSA_EState8', 'CIC2', 'ATSC3d', 'ATSC5p', 'Col_1963', 'ATSC1d', 'SssO', 'Col_106', 'ATSC7d', 'Col_1366', 'Col_81', 'PEOE_VSA2', 'Col_385', 'PEOE_VSA13', 'ATSC7p', 'AATSC0are', 'Col_229', 'JGI8', 'Col_281', 'IC2', 'ATSC7m', 'ATSC1dv', 'IC5', 'ATSC7dv', 'VSA_EState4', 'ATS8m', 'NaasN', 'Col_569', 'SssS', 'ATSC5v', 'PEOE_VSA12', 'Col_1499', 'ATSC4m', 'Col_1041', 'Col_496', 'nHBAcc', 'WPath', 'Col_1118', 'Col_2004', 'PEOE_VSA5', 'MATS1p', 'Col_29', 'Col_1769', 'Col_1142', 'MIC0', 'AATS1are', 'Col_1106', 'AATSC1p', 'JGI9', 'Col_457', 'NaasC', 'Col_725', 'Col_1443', 'Col_1665', 'Col_1284', 'Xp-5dv', 'ATS4m', 'StN', 'AATS1Z', 'Col_1687', 'ATSC8p', 'Col_314', 'Col_1083', 'Col_407', 'Col_1608', 'nAromBond', 'Col_1027', 'SMR_VSA9', 'ATSC7i', 'n9FaRing', 'Col_1115', 'AATS0dv', 'Col_1341', 'ATSC0pe', 'ATSC8dv', 'Col_1604', 'Col_1794', 'Xc-6dv', 'Col_1603', 'SMR_VSA5', 'nG12FaRing', 'PEOE_VSA3', 'Col_1066', 'Col_881', 'Col_1953', 'piPC2', 'n6HRing', 'MZ', 'ATSC0i', 'TSRW10', 'n6aHRing', 'Col_950', 'ATS6dv', 'MIC1', 'nN', 'Col_1060', 'Col_294', 'Col_1017', 'IC1', 'Col_1364', 'Col_772', 'Col_1827', 'Col_123', 'Col_1808', 'Col_1694', 'ATSC2v', 'Col_2033', 'Col_1602', 'Col_284', 'Col_629', 'ATSC4v', 'Col_1903', 'Col_984', 'Col_1971', 'CIC1', 'Col_864', 'Col_1980', 'Col_1148', 'Col_1742', 'Col_21', 'Col_635', 'Col_896', 'Col_875', 'Col_1404', 'Col_1290', 'Col_917', 'Col_1097', 'Col_808', 'Xch-4d', 'Col_103', 'Col_802', 'Col_846', 'Col_1391', 'Col_1855', 'Col_1292', 'Xc-4d', 'Col_1997', 'Col_2009', 'Col_1553', 'Col_1047', 'Col_1917', 'Col_1548', 'Col_956', 'SlogP_VSA11', 'Col_1421', 'Col_1261', 'Col_805', 'nBase', 'Col_768', 'Col_1445', 'Col_878', 'Col_400', 'Col_1731', 'Col_1685', 'Col_1337', 'C3SP3', 'Col_599', 'Col_1550', 'Col_634', 'Col_1812', 'Col_346', 'Col_219', 'nHBDon', 'ATSC8v', 'Col_1119', 'Col_828', 'Col_897', 'Col_351', 'Col_2026', 'Col_1482', 'Col_590', 'Col_1430', 'IC0', 'Col_474', 'Col_1264', 'Col_116', 'Col_1117', 'Col_947', 'Col_769', 'Col_112', 'Col_1116', 'Col_1722', 'Col_145', 'Col_1935', 'Col_395', 'Col_1754', 'Col_80', 'Col_1236', 'Col_1649', 'Col_650', 'Col_1101', 'Col_69', 'Col_898', 'Col_689', 'Col_1440', 'C3SP2', 'Col_1907', 'Col_1696', 'Col_341', 'Col_446', 'n10FHRing', 'nSpiro', 'Col_824', 'Col_1958', 'Col_1139', 'Col_600', 'Col_207', 'Col_114', 'Col_1014', 'Col_237', 'Col_833', 'Col_1015', 'Col_710', 'Col_273', 'Col_203', 'Col_658', 'Col_1547', 'Col_1697', 'Col_473', 'Col_1518', 'Col_62', 'EState_VSA3', 'Col_137', 'Col_2038', 'Col_1265', 'Col_1468', 'Col_1564', 'Col_83', 'Col_56', 'Col_267', 'Col_714', 'Col_192', 'Col_1057', 'Col_456', 'Col_857', 'Col_1525', 'Col_948', 'Col_1968', 'Col_1831', 'Col_184', 'Col_1840', 'Col_1984', 'Col_625', 'Col_1991', 'Col_1607', 'Col_1682', 'Col_1160', 'Col_75', 'nX', 'Col_912', 'Col_201', 'Col_732', 'Col_233', 'Col_930', 'Col_1779', 'StsC', 'nRot', 'EState_VSA10', 'Spe', 'Col_1691', 'Col_102', 'Col_1005', 'Col_822', 'Col_935', 'Col_41', 'Col_245', 'Col_835', 'Col_464', 'n5Ring', 'Col_357', 'Col_2025', 'Col_591', 'Col_302', 'VSA_EState9', 'Col_1', 'Col_310', 'Col_449', 'Col_209', 'Col_1155', 'nO', 'nAHRing', 'TopoShapeIndex', 'Col_1922', 'Col_1114', 'Col_1205', 'Col_1858', 'Col_1145', 'Col_378', 'Col_1999', 'Col_580', 'Col_603', 'Col_922', 'Col_63', 'Col_1111', 'Col_759', 'Col_1891', 'Col_2000', 'Col_1864', 'Col_147', 'Col_584', 'Col_2042', 'Col_1386', 'Col_381', 'Col_666', 'Col_1977', 'Col_892', 'Col_362', 'Col_471', 'Col_1740', 'Col_168', 'Col_1530', 'nG12FHRing', 'Col_392', 'SlogP_VSA4', 'Col_1588', 'Col_2047', 'Col_1074', 'Col_1820', 'Col_1171', 'Col_800', 'Col_598', 'n8FaRing', 'Col_539', 'Col_293', 'Col_1951', 'Col_1357', 'Col_360', 'Col_66', 'SsCl', 'Col_1659', 'Col_1460', 'Col_916', 'Col_1387', 'Col_1905', 'Col_564', 'Col_1593', 'nFRing', 'Col_212', 'Col_1793', 'Col_855', 'Col_544', 'Col_844', 'Col_534', 'Col_333', 'Col_77', 'NsssN', 'SsF', 'Col_1459', 'Col_2044', 'Col_1167', 'NaaaC', 'C1SP2', 'Col_940', 'Col_1034', 'Col_747', 'nS', 'Col_1805', 'Col_1504', 'Col_1733', 'Col_1959', 'Col_1707', 'Col_1889', 'Col_1717', 'Col_1470', 'Col_927', 'Col_504', 'Col_561', 'Col_967', 'Col_1231', 'Col_771', 'Col_718', 'Col_700', 'Col_1356', 'Col_1645', 'Col_486', 'Col_1690', 'Col_1739', 'Col_1981', 'Col_1431', 'Col_1663', 'Col_1379', 'Col_1285', 'NssS', 'Col_752', 'Col_711', 'Col_1121', 'Col_1668', 'Col_1825', 'naHRing', 'n9FAHRing', 'Col_502', 'Col_1489', 'Col_1508', 'Col_2034', 'Col_1759', 'Col_980', 'Col_1966', 'Col_1610', 'Col_1509', 'Col_1565', 'Col_848', 'Col_1993', 'Col_1680', 'Col_1011', 'Col_1817', 'Col_1138', 'Col_1092', 'n10FaHRing', 'Col_1172', 'Col_1237', 'Col_1451', 'Col_1173', 'Col_726', 'Col_708', 'Col_1957', 'Col_1241', 'SsBr', 'Col_667', 'Col_1385', 'Col_739', 'Col_1325', 'Col_1109', 'Col_530', 'Col_1479', 'EState_VSA4', 'Col_1309', 'Col_1979', 'Xc-6d', 'Col_1226', 'Col_1986', 'Col_742', 'Col_915', 'Col_1455', 'Col_1382', 'Col_849', 'nBr', 'Col_1199', 'Col_1010', 'Col_1019', 'Col_1126', 'Col_837', 'Col_1768', 'Col_1693', 'Col_1829', 'Col_525', 'Col_1349', 'Col_1713', 'Col_519', 'Col_1521', 'Col_1318', 'Col_621', 'Col_979', 'Col_1783', 'Col_817', 'Col_1586', 'Col_1128', 'Col_654', 'Col_751', 'Col_547', 'Col_1067', 'Col_1658', 'Col_527', 'Col_996', 'Col_1646', 'Col_1438', 'Col_1990', 'Col_2023', 'n5AHRing', 'nBondsKS', 'Col_932', 'Col_1615', 'Col_942', 'Col_1218', 'Col_2006', 'Col_1800', 'Col_1826', 'Col_867', 'Col_1544', 'Col_2003', 'Col_1238', 'Col_794', 'Col_509', 'Col_1912', 'Col_1496', 'Col_908', 'Col_1481', 'Col_1158', 'Col_1120', 'Col_1747', 'Col_608', 'Col_582', 'Col_1589', 'Col_1896', 'Col_1021', 'n6AHRing', 'n6aRing', 'Col_1582', 'Col_838', 'Col_1089', 'Col_1402', 'Col_1437', 'Col_686', 'Col_1336', 'Col_1801', 'Col_1204', 'Col_1436', 'nFARing', 'Col_1729', 'Col_1426', 'Col_1816', 'Col_1480', 'Col_1235', 'n4Ring', 'Col_1367', 'Col_1785', 'Col_1022', 'Col_2021', 'Col_1970', 'Col_1842', 'Col_993', 'Col_1441', 'n6ARing', 'Col_1897', 'Col_1475', 'Col_1232', 'Col_1096', 'Col_1223', 'Col_1452', 'nHetero', 'Col_1670', 'Col_1181', 'Col_1004', 'Col_1137', 'Col_1012', 'Col_1929', 'Col_1709', 'Col_1698', 'Col_1940', 'Col_1611', 'Col_1281', 'Col_1485', 'Col_1700', 'Col_1370', 'Col_1419', 'Col_1013', 'Col_1432', 'Col_1803', 'Col_1736', 'Col_1312', 'Col_1410', 'Col_1762', 'Col_1295', 'Col_1033', 'n10FAHRing', 'Col_1087', 'Col_1190', 'Col_1642', 'Col_1554', 'Col_1123', 'Col_1280', 'Col_1651', 'nARing', 'Col_1245', 'Col_1775', 'Col_1088', 'Col_1766', 'Col_1982', 'Col_1350', 'n9FRing', 'nP', 'Col_1989', 'Col_1243', 'Col_1375', 'Col_2030', 'Col_1712', 'Col_1862', 'Col_990', 'Col_1577', 'Col_1435', 'Col_1765', 'Col_1843', 'Col_1988', 'Col_1102', 'Col_1039', 'Col_1433', 'Col_1612', 'Col_1321', 'Col_1213', 'Col_1256', 'Col_1664', 'Col_1689', 'Col_1859', 'Col_1983', 'Col_1520', 'Col_1304', 'Col_1276', 'Col_1157', 'Col_1086', 'Col_1149', 'Col_1895', 'Col_1282', 'Col_1778', 'Col_982', 'Col_1154', 'Col_1828', 'Col_2040', 'Col_1405', 'Col_1906', 'Col_1147', 'Col_1136', 'Col_1909', 'Col_1893', 'Col_1073', 'Col_1941', 'Col_1630', 'Col_1730', 'Col_1884', 'Col_1624', 'Col_1676', 'Col_1898', 'Col_1122', 'Col_2024', 'Col_1921', 'Col_1326', 'Col_1985', 'GhoseFilter', 'Col_1998', 'Col_1835', 'Col_1757', 'Col_1599', 'Col_1824', 'n8FRing', 'n5ARing', 'nFaRing', 'Col_1627', 'Col_1622', 'Col_2020', 'Col_1939', 'Col_1601', 'Col_1755', 'Col_1852', 'Col_1791', 'Col_1879', 'Col_1661', 'Col_2001', 'Col_1675', 'Col_1677', 'Col_1620', 'Col_2039', 'Col_1863', 'Col_1894', 'Col_1609', 'Col_1660', 'Col_1576', 'Col_1961', 'Col_1770', 'Col_1773', 'Col_1780', 'Col_1850', 'Col_1927', 'Col_1923', 'Col_1925', 'Col_1542', 'Col_1746', 'Col_1914', 'n3HRing', 'Col_1483', 'n12FHRing', 'Col_2010', 'Col_2032', 'Col_1810', 'Col_1875', 'Col_1901', 'Col_1527', 'Col_1450', 'Col_1948', 'Col_1934', 'Col_1476', 'Col_1574', 'Col_1763', 'Col_1598', 'Col_1681', 'Col_1623', 'Col_1806', 'Col_1528', 'Col_1776', 'Col_1463', 'Col_1946', 'Col_1458', 'Col_1505', 'Col_1473', 'Col_1471', 'Col_1857', 'Col_1796', 'Col_1837', 'Col_2041', 'Col_1882', 'Col_1557', 'Col_1734', 'Col_1799', 'Col_1456', 'Col_1497', 'Col_1637', 'Col_1517', 'Col_1477', 'Col_1851', 'Col_1750', 'Col_1529', 'Col_1807', 'Col_1738', 'Col_1821', 'Col_1880', 'Col_2017', 'Col_1920', 'Col_1861', 'Col_1495', 'Col_1849', 'Col_1978', 'Col_1686', 'Col_1532', 'n12FARing', 'Col_1972', 'Col_1753', 'Col_1802', 'Col_1647', 'Col_2029', 'Col_1618', 'Col_1708', 'Col_1784', 'Col_1626', 'Col_1579', 'Col_1524', 'Col_1911', 'Col_1523', 'Col_1457', 'Col_1652', 'nBridgehead', 'Col_1956', 'nRing', 'Col_2002', 'Col_2022', 'Col_2008', 'Col_1933', 'Col_1952', 'Col_2013', 'Col_1954', 'Col_1975', 'nB', 'Col_2043', 'C2SP1', 'Col_2028', 'Col_1995', 'Col_2019', 'Col_1930', 'nBondsO']



def preprocess_and_scale_Cru(X):
    binary_cols = [col for col in X.columns if 'Col' in col]
    continuous_cols = [col for col in X.columns if col not in binary_cols]
    
    # Use the global scaler object
    X_continuous = X[continuous_cols].values
    X_continuous_scaled = scaler_Cru.transform(X_continuous)  # We'll use transform instead of fit_transform
    
    X_scaled = np.concatenate((X_continuous_scaled, X[binary_cols].values), axis=1)
    return X_scaled

def preprocess_and_scale_Lac(X):
    binary_cols = [col for col in X.columns if 'Col' in col]
    continuous_cols = [col for col in X.columns if col not in binary_cols]
    
    # Use the global scaler object
    X_continuous = X[continuous_cols].values
    X_continuous_scaled = scaler_Lac.transform(X_continuous)  # We'll use transform instead of fit_transform
    
    X_scaled = np.concatenate((X_continuous_scaled, X[binary_cols].values), axis=1)
    return X_scaled



def preprocess_and_scale_Shoi(X):
    binary_cols = [col for col in X.columns if 'Col' in col]
    continuous_cols = [col for col in X.columns if col not in binary_cols]
    
    # Use the global scaler object
    X_continuous = X[continuous_cols].values
    X_continuous_scaled = scaler_Shoi.transform(X_continuous)  # We'll use transform instead of fit_transform
    
    X_scaled = np.concatenate((X_continuous_scaled, X[binary_cols].values), axis=1)
    return X_scaled



def morgan_fpts(data):
    Morgan_fpts = []
    for i in data:
        mol = Chem.MolFromSmiles(i) 
        fpts =  AllChem.GetMorganFingerprintAsBitVect(mol,2,2048)
        mfpts = np.array(fpts)
        Morgan_fpts.append(mfpts)  
    return np.array(Morgan_fpts)



def All_Mordred_descriptors(data):
    calc = Calculator(descriptors, ignore_3D=False)
    mols = [Chem.MolFromSmiles(smi) for smi in data]
    
    # pandas df
    df = calc.pandas(mols)
    return df        

def majority_vote(predictions):
    class_counts = {0: 0, 1: 0}
    for model_prediction in predictions:
        class_counts[model_prediction] += 1

    if class_counts[1] >= 2:
        return 1
    elif class_counts[0] >= 2:
        return 0

def file_download(data,file):
    df = data.to_csv(index=False)
    f= base64.b64encode(df.encode()).decode()
    link = f'<a href ="data:file/csv; base64,{f}" download={file}> Download {file} file</a>'
    return link



def universal_strategy():
  
    st.subheader("Universal Strategy Predictor")

    # Only the Ketcher editor instructions in the sidebar
    st.sidebar.markdown("If you want to draw a molecule, Please use the [**Ketcher editor**](https://lifescience.opensource.epam.com/KetcherDemoSA/index.html) ↗️.")  # The arrow symbol points to the link
    
    # Styling the header for instructions in the sidebar
    st.sidebar.markdown("### Instructions to obtain SMILES from Ketcher:")
    
    # Individual instructions in the sidebar with emphasis on key steps
    st.sidebar.markdown("1. Draw your molecule in the **Ketcher editor**.")
    st.sidebar.markdown("2. Press on the **Save As** option in Ketcher.")
    st.sidebar.markdown("3. Select **SMILES** from the dropdown menu.")
    st.sidebar.markdown("4. **Copy** the provided SMILES representation.")
    st.sidebar.markdown("5. Return to this page and **paste** the SMILES in the box above.")
        
    st.sidebar.write("Molecules Drawing powered by [Ketcher](https://github.com/epam/ketcher).")
    
    # Everything else in the main space
    st.markdown("You can either:")
    st.markdown("1. **Enter your SMILES** in the box below.")
    st.markdown("2. **Upload a CSV file** with the SMILES strings.")
    st.markdown("3. **Draw a Molecule** using Ketcher from the sidebar and obtain the SMILES strings.")
    st.markdown("---")  # Horizontal Line
    
    # Input SMILES strings directly
    one_or_few_SMILES = st.text_input('Enter SMILE Strings in single or double quotation separated by comma:',"['CCCCO']")
    st.markdown('''`or upload SMILE strings in CSV format, note that SMILE strings of the molecules should be in 'SMILES' column:`''')
    
    # File uploader
    many_SMILES = st.file_uploader("Upload your CSV file")
    
    st.markdown("---")  # Horizontal Line
    # Prediction button description and button
    st.markdown("""**If you upload your CSV file or entered the SMILE strings, click the button below to get the Aggregation prediction** """)
    prediction = st.button('Predict Agg status of molecules')



    if one_or_few_SMILES != "['CCCCO']":
        df = pd.DataFrame(eval(one_or_few_SMILES), columns =['SMILES'])
        #========= function call to calculate 200 molecular descriptors using SMILES
        mordred_descriptors_df = All_Mordred_descriptors(df['SMILES'])
        Morgan_fpts = morgan_fpts(df['SMILES'])
        Morgan_fingerprints_df = pd.DataFrame(Morgan_fpts,columns=['Col_{}'.format(i) for i in range(Morgan_fpts.shape[1])])
        exclude_cols = ['Agg status']
        cols_to_add = [col for col in mordred_descriptors_df.columns if col not in exclude_cols]
        Morgan_fingerprints_df[cols_to_add] = mordred_descriptors_df[cols_to_add]
        cat_cols = Morgan_fingerprints_df.select_dtypes(include=['object']).columns.tolist()
        Morgan_fingerprints_df = Morgan_fingerprints_df.drop(cat_cols, axis=1)
        
        #========= Put the 200 molecular descriptors in  data frame
        #========= For Cruzain
        
        X_test_Cruzain = Morgan_fingerprints_df[descriptor_columns_cru]
        X_test_scaled_Cru = preprocess_and_scale_Cru(X_test_Cruzain)
        Agg_Prediction_Cru = model_Cruz.predict(X_test_scaled_Cru)
        
        
        X_test_Lactamase = Morgan_fingerprints_df[descriptor_columns_Lac]
        X_test_scaled_Lac = preprocess_and_scale_Lac(X_test_Lactamase)
        Agg_Prediction_Lac = model_Lac.predict(X_test_scaled_Lac)
        
        
        X_test_Shoichet = Morgan_fingerprints_df[descriptor_columns_Shoi]
        X_test_scaled_Shoi = preprocess_and_scale_Shoi(X_test_Shoichet)
        Agg_Prediction_Shoi = model_Shoichet.predict(X_test_scaled_Shoi) 
        
        majority_predictions = [majority_vote([pred_cru, pred_lac, pred_shoi]) 
                        for pred_cru, pred_lac, pred_shoi in zip(Agg_Prediction_Cru, Agg_Prediction_Lac, Agg_Prediction_Shoi)]

        # Map the predicted values
        status_mapping = {0: "Non-Aggregator", 1: "Aggregator"}
        majority_predictions = [status_mapping[pred] for pred in majority_predictions]
        
        predicted = pd.DataFrame(majority_predictions, columns=['Predicted Agg status'])
        
        output = pd.concat([df, predicted], axis=1)
        
        st.markdown('''## See your output in the following table:''')
        st.write(output)
    
        #======= show CSV file attachment
        st.markdown(file_download(output, "predicted_AggStatus.csv"), unsafe_allow_html=True)
    
    elif prediction:
         df2 = pd.read_csv(many_SMILES)
         #========= function call to calculate 200 molecular descriptors using SMILES
         mordred_descriptors_df = All_Mordred_descriptors(df2['SMILES'])
         Morgan_fpts = morgan_fpts(df2['SMILES'])
         Morgan_fingerprints_df = pd.DataFrame(Morgan_fpts,columns=['Col_{}'.format(i) for i in range(Morgan_fpts.shape[1])])
         exclude_cols = ['Agg status']
         cols_to_add = [col for col in mordred_descriptors_df.columns if col not in exclude_cols]
         Morgan_fingerprints_df[cols_to_add] = mordred_descriptors_df[cols_to_add]
         cat_cols = Morgan_fingerprints_df.select_dtypes(include=['object']).columns.tolist()
         Morgan_fingerprints_df = Morgan_fingerprints_df.drop(cat_cols, axis=1)
         
         #========= Put the 200 molecular descriptors in  data frame
         #========= For Cruzain
         
         X_test_Cruzain = Morgan_fingerprints_df[descriptor_columns_cru]
         X_test_scaled_Cru = preprocess_and_scale_Cru(X_test_Cruzain)
         Agg_Prediction_Cru = model_Cruz.predict(X_test_scaled_Cru)
         
         
         X_test_Lactamase = Morgan_fingerprints_df[descriptor_columns_Lac]
         X_test_scaled_Lac = preprocess_and_scale_Lac(X_test_Lactamase)
         Agg_Prediction_Lac = model_Lac.predict(X_test_scaled_Lac)
         
         
         X_test_Shoichet = Morgan_fingerprints_df[descriptor_columns_Shoi]
         X_test_scaled_Shoi = preprocess_and_scale_Shoi(X_test_Shoichet)
         Agg_Prediction_Shoi = model_Shoichet.predict(X_test_scaled_Shoi) 
         
         majority_predictions = [majority_vote([pred_cru, pred_lac, pred_shoi]) 
                        for pred_cru, pred_lac, pred_shoi in zip(Agg_Prediction_Cru, Agg_Prediction_Lac, Agg_Prediction_Shoi)]

          # Map the predicted values
          status_mapping = {0: "Non-Aggregator", 1: "Aggregator"}
          majority_predictions = [status_mapping[pred] for pred in majority_predictions]
          
          predicted = pd.DataFrame(majority_predictions, columns=['Predicted Agg status'])
          
          output = pd.concat([df2, predicted], axis=1)
          
          st.markdown('''## See your output in the following table:''')
          st.write(output)
    
         #======= show CSV file attachment
         st.markdown(file_download(output, "predicted_AggStatus.csv"), unsafe_allow_html=True)


    
# Page Configuration
# Page Configuration
st.set_page_config(page_title='Aggregation Prediction App', layout='wide')

# Title
st.title('An Accurate Predictor for Colloidal Aggregation')

# Navigation sidebar
st.sidebar.markdown('<h2 style="color:#5a03fc;background-color:powderblue;border-radius:10px;text-align:center"> Welcome to CAP! </h2>',unsafe_allow_html=True)
page = st.sidebar.selectbox("Choose from here:", ["Home", "Prediction Tool"])
st.sidebar.markdown("""
Use the dropdown above to navigate between the home page and the prediction tool. 
Whether you're new or returning, we're here to help you assess the aggregation behavior of your molecules.
""")

if page == "Home":
    home_page()
elif page == "Prediction Tool":
    universal_strategy()

        
