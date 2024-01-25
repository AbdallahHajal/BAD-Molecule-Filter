from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from sklearn.preprocessing import StandardScaler
from pyADA import ApplicabilityDomain
import os
import pickle
from sklearn.metrics.pairwise import pairwise_distances
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import pandas as pd
from io import BytesIO
import tempfile
import uuid



# Initialize Flask app
app = Flask(__name__)

# Cache for models, scalers, and training sets
resource_cache = {}

def load_resource_lazy(file_path):
    if file_path not in resource_cache:
        try:
            with open(file_path, 'rb') as f:
                resource_cache[file_path] = pickle.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None
    return resource_cache[file_path]



# Preprocessing functions
def preprocess_and_scale_Cru(X,scaler_CRU):
    binary_cols = [col for col in X.columns if 'Col' in col]
    continuous_cols = [col for col in X.columns if col not in binary_cols]
    
    # Use the global scaler object
    X_continuous = X[continuous_cols].values
    X_continuous_scaled = scaler_CRU.transform(X_continuous)  # We'll use transform instead of fit_transform
    
    X_scaled = np.concatenate((X_continuous_scaled, X[binary_cols].values), axis=1)
    return X_scaled

def preprocess_and_scale_Lac(X,scaler_Lac):
    binary_cols = [col for col in X.columns if 'Col' in col]
    continuous_cols = [col for col in X.columns if col not in binary_cols]
    
    # Use the global scaler object
    X_continuous = X[continuous_cols].values
    X_continuous_scaled = scaler_Lac.transform(X_continuous)  # We'll use transform instead of fit_transform
    
    X_scaled = np.concatenate((X_continuous_scaled, X[binary_cols].values), axis=1)
    return X_scaled

def preprocess_and_scale_Sho(X,scaler_Shoi):
    binary_cols = [col for col in X.columns if 'Col' in col]
    continuous_cols = [col for col in X.columns if col not in binary_cols]
    
    # Use the global scaler object
    X_continuous = X[continuous_cols].values
    X_continuous_scaled = scaler_Shoi.transform(X_continuous)  # We'll use transform instead of fit_transform
    
    X_scaled = np.concatenate((X_continuous_scaled, X[binary_cols].values), axis=1)
    return X_scaled
def preprocess_and_scale_MM(X,scaler_MM):
    binary_cols = [col for col in X.columns if 'Col' in col]
    continuous_cols = [col for col in X.columns if col not in binary_cols]
    
    # Use the global scaler object
    X_continuous = X[continuous_cols].values
    X_continuous_scaled = scaler_MM.transform(X_continuous)  # We'll use transform instead of fit_transform
    
    X_scaled = np.concatenate((X_continuous_scaled, X[binary_cols].values), axis=1)
    return X_scaled
# Descriptors functions
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
start_column_Cru = 163
end_column_Cru = 163

descriptor_columns_Lac=['MWC01', 'nHBDon', 'SLogP', 'ATS0p', 'SlogP_VSA2', 'FilterItLogS', 'nBondsD', 'VSA_EState8', 'Col_2009', 'fragCpx','GGI9', 'ATSC1i', 'piPC9', 'SdO', 'Col_1602', 'nX', 'SMR_VSA3', 'PEOE_VSA8', 'ATSC3i', 'TMPC10', 'FCSP3', 'MATS1v', 'ATSC0i', 'ATS5p', 'AATSC1pe', 'ATSC2v', 'EState_VSA9', 'ATSC2i', 'ATS7m', 'nAcid', 'piPC1', 'Col_216', 'TopoPSA', 'NdS', 'Col_1849', 'MIC3', 'SlogP_VSA3', 'SlogP_VSA7','VSA_EState3', 'WPath', 'VSA_EState2', 'Col_798', 'ATS0m','ATSC3dv', 'Col_738','Col_1549', 'PEOE_VSA6', 'nBase', 'Col_1417', 'SMR_VSA6', 'Col_714', 'PEOE_VSA9', 'SaaO', 'Col_1089', 'Col_1160', 'IC2', 'ATSC1Z', 'SsssN', 'AATS1i', 'TSRW10', 'Col_222', 'SsCl', 'SlogP_VSA8', 'C2SP2', 'Col_148', 'BalabanJ', 'AATSC0p', 'IC5', 'ATSC6i', 'Col_1604', 'SlogP_VSA1', 'Col_277', 'Col_74', 'Col_1816', 'Col_314', 'ATSC5i', 'AATS1v', 'Xch-7d', 'NaaN', 'ATSC8i', 'GATS1i', 'PEOE_VSA3', 'Col_378','Col_473', 'ATSC3Z', 'PEOE_VSA13', 'VSA_EState5', 'GATS1m',  'Col_401', 'SsssNH', 'ATSC4dv', 'Xp-3dv', 'ATSC2dv','Col_50', 'Col_495', 'Xch-6dv', 'Xc-5d', 'AATS0p', 'nBondsA', 'Col_1804', 'ATSC4Z', 'TopoPSA(NO)', 'VSA_EState9', 'Col_273', 'Col_1825', 'SdssC', 'GATS1v', 'ATSC4d', 'JGI6', 'Col_1603', 'Col_1653', 'Col_949', 'SdsssP', 'ATSC1dv', 'Col_695', 'PEOE_VSA11', 'SMR_VSA9', 'ATSC7p','ATS3Z', 'IC0', 'Col_1250', 'MATS1dv', 'ZMIC1', 'ATSC5d', 'Col_942',  'Col_639', 'Col_831', 'Col_1508', 'Col_12', 'Col_82', 'ATSC3d', 'ATSC5Z', 'Col_423', 'JGI5', 'Col_2038', 'Col_532', 'Col_656', 'JGI4', 'Col_809', 'PEOE_VSA4', 'Col_145', 'PEOE_VSA1', 'SaasN', 'Col_1979', 'Col_586', 'Col_155', 'Col_1389', 'Col_167', 'ATSC6d', 'C3SP2', 'ATSC4v', 'Col_522', 'Col_1041', 'ATSC8are', 'ATSC1d', 'Col_52', 'Col_1217', 'PEOE_VSA12', 'JGI2', 'Col_1918', 'fMF', 'VSA_EState7', 'Xch-7dv',  'SMR_VSA1', 'Col_521', 'Col_41', 'Xpc-5d', 'PEOE_VSA10', 'Col_1780','SssssC', 'Col_1504', 'Col_1130', 'ATSC5v', 'ATSC7v', 'n5aRing', 'ATSC6dv', 'Col_405', 'Col_1934',  'Col_42', 'Col_980', 'Col_1416', 'ATSC8p', 'Col_396', 'piPC3', 'Col_239', 'ATSC5p', 'Col_845', 'ATSC8d', 'Col_1135', 'Col_679', 'Col_1409', 'Col_251', 'Col_233', 'Col_39', 'nSpiro', 'Col_1861', 'AATSC0dv', 'Col_602', 'JGI8', 'MATS1p', 'Col_1014', 'ATSC4are', 'Col_974', 'ATSC3v', 'GATS1dv', 'AATSC1d', 'ATSC2d', 'PEOE_VSA2', 'Col_268', 'CIC4', 'MATS1are', 'ATSC0p', 'ECIndex', 'Col_490', 'Col_783', 'Col_1430', 'ATSC7i', 'Col_1097', 'Col_92', 'Col_1177', 'Col_1761', 'ZMIC3', 'Col_1622', 'Col_1342', 'ATS8dv', 'Col_433', 'Col_1488', 'Col_106', 'ATSC7dv', 'Col_2020', 'Col_623', 'ATSC7Z',  'Col_278', 'Col_515', 'Col_1329', 'C1SP2', 'Col_1245', 'Col_1624', 'Col_636', 'GGI1', 'Col_597', 'EState_VSA8', 'Col_247', 'SlogP_VSA10', 'Col_1145', 'SMR_VSA5', 'Col_576', 'Col_504', 'ATSC0pe', 'Col_94', 'Col_162', 'NssO', 'Col_789', 'GATS1p', 'IC1', 'Col_13', 'SsBr', 'Col_1022', 'Xc-3d', 'PEOE_VSA7', 'Col_1889', 'SlogP_VSA11', 'Col_420', 'ATSC8m', 'Col_1065', 'SaasC', 'Col_1310', 'ATSC5dv', 'Col_1084', 'RotRatio', 'Col_1964', 'SIC0', 'Col_140', 'Col_1660', 'NdsN', 'Col_1845', 'Col_417', 'Col_1453', 'Col_1866', 'Col_245', 'Col_227', 'Col_1012', 'Col_506', 'Col_1047', 'Col_1063', 'Col_923', 'Col_1944', 'Col_17', 'Col_645', 'Col_1142', 'Col_207', 'Col_1806',  'ATSC6p', 'Col_7', 'JGI10', 'Col_885', 'Col_611', 'AATS1Z','Col_1517', 'Col_369', 'Col_1341', 'Col_451', 'Col_397', 'Diameter', 'Xpc-6dv', 'Col_2041', 'Col_950', 'Col_1260', 'AATSC0Z', 'Col_771', 'Col_1385', 'Col_113', 'SlogP_VSA4', 'Col_1254', 'ATSC8dv', 'ATSC6Z', 'Col_26', 'Col_1353', 'Col_1411', 'Col_781', 'Col_779', 'Col_1444', 'VSA_EState4', 'Col_867', 'CIC1', 'Col_30','ATSC4p', 'Col_334', 'EState_VSA10', 'Col_1835', 'Col_1356', 'MWC04', 'nRot', 'Col_1634', 'Col_1449', 'Col_255', 'Col_457', 'Col_487', 'Col_965', 'Col_2006', 'Col_2013', 'Col_951', 'Col_1850', 'Mi', 'Col_1966', 'Col_1765', 'Col_1395', 'Col_1006', 'ATSC7d', 'Col_339', 'Col_1921', 'nHetero', 'Col_1357', 'ATSC6v', 'Col_1847', 'Xch-5d', 'Col_1750', 'ATS2dv', 'Col_561', 'Col_1295', 'Col_102', 'Col_667', 'Col_80', 'ATS6dv', 'ATSC6pe', 'Col_1476', 'Col_863', 'Col_1911', 'Col_901', 'ATSC4i', 'Col_896', 'Col_893', 'AATSC0v', 'Col_1425', 'SssS', 'Col_1920', 'nFARing', 'Col_1611', 'Col_1694', 'NaasC', 'Col_620', 'AATSC0i', 'Col_1281', 'MIC0', 'Col_832', 'Col_1363', 'Col_1261', 'Col_1775', 'Col_1317', 'Xc-4dv', 'n9FaRing', 'Col_1490', 'JGI7', 'Col_1617', 'Col_1121', 'ATSC3p', 'Col_1060', 'SlogP_VSA6', 'Col_1656', 'Col_488', 'EState_VSA2', 'n9FRing', 'Col_670', 'Col_1695', 'Col_808', 'Col_1481','Col_1312', 'Col_1701', 'Col_613', 'Col_784', 'Xch-6d', 'Col_1612', 'n5Ring', 'Col_1627', 'Col_1055', 'Col_1163', 'Col_1897', 'Col_1445', 'C4SP3', 'Col_2044', 'nHRing', 'VSA_EState6', 'Col_1296', 'Col_1736', 'Col_826', 'Col_1802', 'Col_1589', 'SaaS', 'Col_1017', 'Col_1152', 'Col_1513', 'Col_961', 'SaaaC', 'Col_1675', 'JGI3', 'Col_496', 'Col_1309', 'Col_844', 'Col_823', 'Col_1930', 'Col_1967', 'SlogP_VSA5', 'Col_1469', 'GGI3', 'SMR_VSA4', 'Col_618', 'Col_1716', 'Col_1745', 'JGI9', 'NaasN', 'Col_1600', 'Col_1005', 'Xc-3dv', 'Col_1387', 'Col_1077', 'Col_1322', 'n6aHRing', 'ATSC0d', 'Col_1212', 'Col_637', 'Col_1238', 'Col_1522', 'Xc-4d', 'Col_849', 'Col_1833', 'Col_1514', 'ZMIC2', 'Col_540', 'Col_1667', 'Col_1932', 'TopoShapeIndex', 'Col_1304', 'Col_2028', 'Col_1637', 'Col_1087', 'Col_1057', 'Col_1034', 'Col_1638', 'Col_1905', 'Col_1749', 'Col_1298', 'Col_1482', 'Col_1757', 'Col_1741', 'Col_1655', 'Col_1848', 'Col_1792', 'Col_2029', 'Col_1235', 'Col_1114', 'n6HRing', 'Col_1105', 'Col_1981', 'Col_1463', 'Col_1115', 'AATSC0d', 'Col_1362', 'Xc-5dv', 'Col_1722', 'Col_1032', 'Col_1129', 'PEOE_VSA5', 'Col_1713', 'Col_1904', 'Col_1498', 'MZ', 'Col_1516', 'Col_1033', 'Col_1810', 'Col_1950', 'CIC2', 'Mpe', 'Col_1754', 'Col_1914', 'Col_1050', 'Col_1519', 'Col_1443', 'n6ARing', 'n4aRing', 'Col_1291', 'ATSC8v', 'Col_1240', 'SMR_VSA2', 'Col_1000', 'Col_1586', 'Col_996', 'n5AHRing', 'Col_1592', 'Col_1308', 'Col_1194', 'Col_1441', 'Col_1860', 'Col_1193', 'Col_1676', 'Col_1405', 'EState_VSA4', 'AATSC0are', 'Col_1066', 'Col_1480', 'Col_1956', 'NsssNH', 'Col_1791', 'nG12FAHRing', 'Col_1783', 'Col_1998','Xch-5dv', 'GhoseFilter', 'Col_1705', 'Col_2025', 'Col_2017', 'Col_1620', 'Col_2039', 'n6Ring', 'Col_1855', 'Col_1630', 'Col_1719', 'Col_1668', 'SRW09', 'StsC', 'Col_1503', 'Col_1601', 'nN', 'nARing', 'Col_1925', 'Xc-6dv', 'Col_1560', 'Col_1768', 'Col_1731', 'Col_1984', 'ATSC2m', 'Col_1585', 'Col_1865', 'Col_1610', 'n5ARing', 'Col_1827', 'Col_1727', 'Col_1698', 'Xc-6d', 'Col_1641', 'C3SP3', 'Col_1779', 'nO', 'Col_674', 'Col_1970', 'Col_1959', 'Col_1971', 'n12FAHRing', 'n6aRing', 'Col_2042', 'EState_VSA7', 'Col_1975', 'nS', 'Xch-4d', 'Col_1982', 'Col_2001']
start_column_Lac= 250
end_column_Lac = 250

descriptor_columns_Sho=['SLogP', 'FilterItLogS', 'SddsN', 'MATS1v', 'AATS1p', 'SaaN', 'nBondsD', 'MATS1i', 'SaaO', 'C2SP2', 'Col_694', 'Diameter', 'Col_1867', 'NdsN', 'SMR_VSA3', 'SlogP_VSA7', 'SlogP_VSA1', 'PEOE_VSA4', 'GATS1p', 'Col_110', 'Lipinski', 'GATS1v', 'AATS1i', 'SlogP_VSA2', 'EState_VSA8', 'nAcid', 'BalabanJ', 'ATSC3are', 'VSA_EState3', 'GATS1i', 'GATS1are', 'GGI10', 'VSA_EState6', 'JGI1', 'ATSC5dv', 'Mp', 'PEOE_VSA9', 'Xch-6d', 'n4aRing', 'PEOE_VSA7', 'piPC10', 'ZMIC1', 'AATSC0Z', 'MIC5', 'AATS0i', 'AATSC1are', 'SdO', 'SMR_VSA7', 'Col_31', 'ATSC4are', 'NdS', 'GATS1dv', 'AATSC1Z', 'Col_1449', 'ATSC2m', 'MATS1Z', 'SlogP_VSA3', 'SaaaC', 'VSA_EState2', 'AATSC0d', 'n5aHRing', 'Col_656', 'JGI6', 'SdssC', 'nHRing', 'SMR_VSA2', 'ATSC2d', 'Xch-6dv', 'ATSC8are', 'GGI5', 'AATS1dv', 'PEOE_VSA10', 'ATS0m', 'GGI3', 'ATSC5i', 'EState_VSA2', 'ATSC7v', 'SlogP_VSA10', 'SMR_VSA4', 'ATS3dv', 'MW', 'ECIndex', 'ATSC6d', 'ATSC3dv', 'JGI5', 'SRW09', 'PEOE_VSA11', 'ATSC4d', 'Col_1745', 'JGI2', 'AATSC0v', 'Col_1195', 'Xp-2dv', 'ATSC4p', 'ZMIC4', 'Col_162', 'VSA_EState5', 'SssssC', 'Xp-5d', 'Col_1866', 'ATSC4i', 'NsssNH', 'Xpc-4dv', 'ATSC2i', 'SsssN', 'RotRatio', 'PEOE_VSA8', 'AATSC0dv', 'CIC4', 'SlogP_VSA8', 'AATSC1dv', 'ATS6v', 'SMR_VSA1', 'Xc-4dv', 'Xpc-4d', 'Xc-5dv', 'Col_1749', 'ATS8dv', 'Col_1038', 'ATSC7pe', 'VSA_EState7', 'ATSC3p', 'ATSC3Z', 'PEOE_VSA1', 'ATS0p', 'Xc-3d', 'ATSC6pe', 'AATS0p', 'SaasN', 'ATSC8i', 'EState_VSA9', 'fMF', 'MPC10', 'Col_1210', 'ATSC6m', 'Col_1823', 'CIC0', 'AATSC0i', 'Col_750', 'PEOE_VSA6', 'Col_859', 'ATSC8m', 'GATS1m', 'Xc-3dv', 'Col_45', 'SMR_VSA6', 'ZMIC2', 'ATSC6v', 'SsssNH', 'ATSC5pe', 'Col_1331', 'AATSC1d', 'JGI10', 'ATSC4dv', 'SaasC', 'Xch-5dv', 'Col_1947', 'JGI7', 'ATSC8d', 'ATSC6i', 'TopoPSA', 'ATSC6dv', 'NaaS', 'ATSC6p', 'Xch-5d', 'ATSC5m', 'JGI4', 'Col_1847', 'ATSC2dv', 'Col_1174', 'Col_167', 'SlogP_VSA5', 'AATS0d', 'Xc-5d', 'ATSC2p', 'n6Ring', 'TopoPSA(NO)', 'ATSC2are', 'JGI3', 'Col_1472', 'Col_1416', 'Col_1316', 'Col_617', 'Col_264', 'ATSC5d', 'Col_753', 'SaaS', 'ATSC3i', 'ATSC3v', 'VSA_EState8', 'CIC2', 'ATSC3d', 'ATSC5p', 'Col_1963', 'ATSC1d', 'SssO', 'Col_106', 'ATSC7d', 'Col_1366', 'Col_81', 'PEOE_VSA2', 'Col_385', 'PEOE_VSA13', 'ATSC7p', 'AATSC0are', 'Col_229', 'JGI8', 'Col_281', 'IC2', 'ATSC7m', 'ATSC1dv', 'IC5', 'ATSC7dv', 'VSA_EState4', 'ATS8m', 'NaasN', 'Col_569', 'SssS', 'ATSC5v', 'PEOE_VSA12', 'Col_1499', 'ATSC4m', 'Col_1041', 'Col_496', 'nHBAcc', 'WPath', 'Col_1118', 'Col_2004', 'PEOE_VSA5', 'MATS1p', 'Col_29', 'Col_1769', 'Col_1142', 'MIC0', 'AATS1are', 'Col_1106', 'AATSC1p', 'JGI9', 'Col_457', 'NaasC', 'Col_725', 'Col_1443', 'Col_1665', 'Col_1284', 'Xp-5dv', 'ATS4m', 'StN', 'AATS1Z', 'Col_1687', 'ATSC8p', 'Col_314', 'Col_1083', 'Col_407', 'Col_1608', 'nAromBond', 'Col_1027', 'SMR_VSA9', 'ATSC7i', 'n9FaRing', 'Col_1115', 'AATS0dv', 'Col_1341', 'ATSC0pe', 'ATSC8dv', 'Col_1604', 'Col_1794', 'Xc-6dv', 'Col_1603', 'SMR_VSA5', 'nG12FaRing', 'PEOE_VSA3', 'Col_1066', 'Col_881', 'Col_1953', 'piPC2', 'n6HRing', 'MZ', 'ATSC0i', 'TSRW10', 'n6aHRing', 'Col_950', 'ATS6dv', 'MIC1', 'nN', 'Col_1060', 'Col_294', 'Col_1017', 'IC1', 'Col_1364', 'Col_772', 'Col_1827', 'Col_123', 'Col_1808', 'Col_1694', 'ATSC2v', 'Col_2033', 'Col_1602', 'Col_284', 'Col_629', 'ATSC4v', 'Col_1903', 'Col_984', 'Col_1971', 'CIC1', 'Col_864', 'Col_1980', 'Col_1148', 'Col_1742', 'Col_21', 'Col_635', 'Col_896', 'Col_875', 'Col_1404', 'Col_1290', 'Col_917', 'Col_1097', 'Col_808', 'Xch-4d', 'Col_103', 'Col_802', 'Col_846', 'Col_1391', 'Col_1855', 'Col_1292', 'Xc-4d', 'Col_1997', 'Col_2009', 'Col_1553', 'Col_1047', 'Col_1917', 'Col_1548', 'Col_956', 'SlogP_VSA11', 'Col_1421', 'Col_1261', 'Col_805', 'nBase', 'Col_768', 'Col_1445', 'Col_878', 'Col_400', 'Col_1731', 'Col_1685', 'Col_1337', 'C3SP3', 'Col_599', 'Col_1550', 'Col_634', 'Col_1812', 'Col_346', 'Col_219', 'nHBDon', 'ATSC8v', 'Col_1119', 'Col_828', 'Col_897', 'Col_351', 'Col_2026', 'Col_1482', 'Col_590', 'Col_1430', 'IC0', 'Col_474', 'Col_1264', 'Col_116', 'Col_1117', 'Col_947', 'Col_769', 'Col_112', 'Col_1116', 'Col_1722', 'Col_145', 'Col_1935', 'Col_395', 'Col_1754', 'Col_80', 'Col_1236', 'Col_1649', 'Col_650', 'Col_1101', 'Col_69', 'Col_898', 'Col_689', 'Col_1440', 'C3SP2', 'Col_1907', 'Col_1696', 'Col_341', 'Col_446', 'n10FHRing', 'nSpiro', 'Col_824', 'Col_1958', 'Col_1139', 'Col_600', 'Col_207', 'Col_114', 'Col_1014', 'Col_237', 'Col_833', 'Col_1015', 'Col_710', 'Col_273', 'Col_203', 'Col_658', 'Col_1547', 'Col_1697', 'Col_473', 'Col_1518', 'Col_62', 'EState_VSA3', 'Col_137', 'Col_2038', 'Col_1265', 'Col_1468', 'Col_1564', 'Col_83', 'Col_56', 'Col_267', 'Col_714', 'Col_192', 'Col_1057', 'Col_456', 'Col_857', 'Col_1525', 'Col_948', 'Col_1968', 'Col_1831', 'Col_184', 'Col_1840', 'Col_1984', 'Col_625', 'Col_1991', 'Col_1607', 'Col_1682', 'Col_1160', 'Col_75', 'nX', 'Col_912', 'Col_201', 'Col_732', 'Col_233', 'Col_930', 'Col_1779', 'StsC', 'nRot', 'EState_VSA10', 'Spe', 'Col_1691', 'Col_102', 'Col_1005', 'Col_822', 'Col_935', 'Col_41', 'Col_245', 'Col_835', 'Col_464', 'n5Ring', 'Col_357', 'Col_2025', 'Col_591', 'Col_302', 'VSA_EState9', 'Col_1', 'Col_310', 'Col_449', 'Col_209', 'Col_1155', 'nO', 'nAHRing', 'TopoShapeIndex', 'Col_1922', 'Col_1114', 'Col_1205', 'Col_1858', 'Col_1145', 'Col_378', 'Col_1999', 'Col_580', 'Col_603', 'Col_922', 'Col_63', 'Col_1111', 'Col_759', 'Col_1891', 'Col_2000', 'Col_1864', 'Col_147', 'Col_584', 'Col_2042', 'Col_1386', 'Col_381', 'Col_666', 'Col_1977', 'Col_892', 'Col_362', 'Col_471', 'Col_1740', 'Col_168', 'Col_1530', 'nG12FHRing', 'Col_392', 'SlogP_VSA4', 'Col_1588', 'Col_2047', 'Col_1074', 'Col_1820', 'Col_1171', 'Col_800', 'Col_598', 'n8FaRing', 'Col_539', 'Col_293', 'Col_1951', 'Col_1357', 'Col_360', 'Col_66', 'SsCl', 'Col_1659', 'Col_1460', 'Col_916', 'Col_1387', 'Col_1905', 'Col_564', 'Col_1593', 'nFRing', 'Col_212', 'Col_1793', 'Col_855', 'Col_544', 'Col_844', 'Col_534', 'Col_333', 'Col_77', 'NsssN', 'SsF', 'Col_1459', 'Col_2044', 'Col_1167', 'NaaaC', 'C1SP2', 'Col_940', 'Col_1034', 'Col_747', 'nS', 'Col_1805', 'Col_1504', 'Col_1733', 'Col_1959', 'Col_1707', 'Col_1889', 'Col_1717', 'Col_1470', 'Col_927', 'Col_504', 'Col_561', 'Col_967', 'Col_1231', 'Col_771', 'Col_718', 'Col_700', 'Col_1356', 'Col_1645', 'Col_486', 'Col_1690', 'Col_1739', 'Col_1981', 'Col_1431', 'Col_1663', 'Col_1379', 'Col_1285', 'NssS', 'Col_752', 'Col_711', 'Col_1121', 'Col_1668', 'Col_1825', 'naHRing', 'n9FAHRing', 'Col_502', 'Col_1489', 'Col_1508', 'Col_2034', 'Col_1759', 'Col_980', 'Col_1966', 'Col_1610', 'Col_1509', 'Col_1565', 'Col_848', 'Col_1993', 'Col_1680', 'Col_1011', 'Col_1817', 'Col_1138', 'Col_1092', 'n10FaHRing', 'Col_1172', 'Col_1237', 'Col_1451', 'Col_1173', 'Col_726', 'Col_708', 'Col_1957', 'Col_1241', 'SsBr', 'Col_667', 'Col_1385', 'Col_739', 'Col_1325', 'Col_1109', 'Col_530', 'Col_1479', 'EState_VSA4', 'Col_1309', 'Col_1979', 'Xc-6d', 'Col_1226', 'Col_1986', 'Col_742', 'Col_915', 'Col_1455', 'Col_1382', 'Col_849', 'nBr', 'Col_1199', 'Col_1010', 'Col_1019', 'Col_1126', 'Col_837', 'Col_1768', 'Col_1693', 'Col_1829', 'Col_525', 'Col_1349', 'Col_1713', 'Col_519', 'Col_1521', 'Col_1318', 'Col_621', 'Col_979', 'Col_1783', 'Col_817', 'Col_1586', 'Col_1128', 'Col_654', 'Col_751', 'Col_547', 'Col_1067', 'Col_1658', 'Col_527', 'Col_996', 'Col_1646', 'Col_1438', 'Col_1990', 'Col_2023', 'n5AHRing', 'nBondsKS', 'Col_932', 'Col_1615', 'Col_942', 'Col_1218', 'Col_2006', 'Col_1800', 'Col_1826', 'Col_867', 'Col_1544', 'Col_2003', 'Col_1238', 'Col_794', 'Col_509', 'Col_1912', 'Col_1496', 'Col_908', 'Col_1481', 'Col_1158', 'Col_1120', 'Col_1747', 'Col_608', 'Col_582', 'Col_1589', 'Col_1896', 'Col_1021', 'n6AHRing', 'n6aRing', 'Col_1582', 'Col_838', 'Col_1089', 'Col_1402', 'Col_1437', 'Col_686', 'Col_1336', 'Col_1801', 'Col_1204', 'Col_1436', 'nFARing', 'Col_1729', 'Col_1426', 'Col_1816', 'Col_1480', 'Col_1235', 'n4Ring', 'Col_1367', 'Col_1785', 'Col_1022', 'Col_2021', 'Col_1970', 'Col_1842', 'Col_993', 'Col_1441', 'n6ARing', 'Col_1897', 'Col_1475', 'Col_1232', 'Col_1096', 'Col_1223', 'Col_1452', 'nHetero', 'Col_1670', 'Col_1181', 'Col_1004', 'Col_1137', 'Col_1012', 'Col_1929', 'Col_1709', 'Col_1698', 'Col_1940', 'Col_1611', 'Col_1281', 'Col_1485', 'Col_1700', 'Col_1370', 'Col_1419', 'Col_1013', 'Col_1432', 'Col_1803', 'Col_1736', 'Col_1312', 'Col_1410', 'Col_1762', 'Col_1295', 'Col_1033', 'n10FAHRing', 'Col_1087', 'Col_1190', 'Col_1642', 'Col_1554', 'Col_1123', 'Col_1280', 'Col_1651', 'nARing', 'Col_1245', 'Col_1775', 'Col_1088', 'Col_1766', 'Col_1982', 'Col_1350', 'n9FRing', 'nP', 'Col_1989', 'Col_1243', 'Col_1375', 'Col_2030', 'Col_1712', 'Col_1862', 'Col_990', 'Col_1577', 'Col_1435', 'Col_1765', 'Col_1843', 'Col_1988', 'Col_1102', 'Col_1039', 'Col_1433', 'Col_1612', 'Col_1321', 'Col_1213', 'Col_1256', 'Col_1664', 'Col_1689', 'Col_1859', 'Col_1983', 'Col_1520', 'Col_1304', 'Col_1276', 'Col_1157', 'Col_1086', 'Col_1149', 'Col_1895', 'Col_1282', 'Col_1778', 'Col_982', 'Col_1154', 'Col_1828', 'Col_2040', 'Col_1405', 'Col_1906', 'Col_1147', 'Col_1136', 'Col_1909', 'Col_1893', 'Col_1073', 'Col_1941', 'Col_1630', 'Col_1730', 'Col_1884', 'Col_1624', 'Col_1676', 'Col_1898', 'Col_1122', 'Col_2024', 'Col_1921', 'Col_1326', 'Col_1985', 'GhoseFilter', 'Col_1998', 'Col_1835', 'Col_1757', 'Col_1599', 'Col_1824', 'n8FRing', 'n5ARing', 'nFaRing', 'Col_1627', 'Col_1622', 'Col_2020', 'Col_1939', 'Col_1601', 'Col_1755', 'Col_1852', 'Col_1791', 'Col_1879', 'Col_1661', 'Col_2001', 'Col_1675', 'Col_1677', 'Col_1620', 'Col_2039', 'Col_1863', 'Col_1894', 'Col_1609', 'Col_1660', 'Col_1576', 'Col_1961', 'Col_1770', 'Col_1773', 'Col_1780', 'Col_1850', 'Col_1927', 'Col_1923', 'Col_1925', 'Col_1542', 'Col_1746', 'Col_1914', 'n3HRing', 'Col_1483', 'n12FHRing', 'Col_2010', 'Col_2032', 'Col_1810', 'Col_1875', 'Col_1901', 'Col_1527', 'Col_1450', 'Col_1948', 'Col_1934', 'Col_1476', 'Col_1574', 'Col_1763', 'Col_1598', 'Col_1681', 'Col_1623', 'Col_1806', 'Col_1528', 'Col_1776', 'Col_1463', 'Col_1946', 'Col_1458', 'Col_1505', 'Col_1473', 'Col_1471', 'Col_1857', 'Col_1796', 'Col_1837', 'Col_2041', 'Col_1882', 'Col_1557', 'Col_1734', 'Col_1799', 'Col_1456', 'Col_1497', 'Col_1637', 'Col_1517', 'Col_1477', 'Col_1851', 'Col_1750', 'Col_1529', 'Col_1807', 'Col_1738', 'Col_1821', 'Col_1880', 'Col_2017', 'Col_1920', 'Col_1861', 'Col_1495', 'Col_1849', 'Col_1978', 'Col_1686', 'Col_1532', 'n12FARing', 'Col_1972', 'Col_1753', 'Col_1802', 'Col_1647', 'Col_2029', 'Col_1618', 'Col_1708', 'Col_1784', 'Col_1626', 'Col_1579', 'Col_1524', 'Col_1911', 'Col_1523', 'Col_1457', 'Col_1652', 'nBridgehead', 'Col_1956', 'nRing', 'Col_2002', 'Col_2022', 'Col_2008', 'Col_1933', 'Col_1952', 'Col_2013', 'Col_1954', 'Col_1975', 'nB', 'Col_2043', 'C2SP1', 'Col_2028', 'Col_1995', 'Col_2019', 'Col_1930', 'nBondsO']
start_column_Shoi= 287
end_column_Shoi = 287

descriptor_columns_MM=['ATS0p', 'SLogP', 'ATSC1Z', 'Col_694', 'TopoPSA', 'nAcid', 'piPC10', 'SlogP_VSA2', 'nHBDon', 'AATSC1are', 'BalabanJ', 'SsssNH', 'ZMIC0', 'Col_2009', 'ATSC0v', 'SlogP_VSA1', 'n6aHRing', 'MPC9', 'SdO', 'ECIndex', 'SMR_VSA1', 'NdsN', 'Col_1602', 'VSA_EState8', 'AATSC1v', 'SssS', 'EState_VSA9', 'SMR_VSA6', 'FilterItLogS', 'SaaN', 'Col_222', 'ATSC2p', 'WPath', 'Col_516', 'C3SP2', 'AATS1i', 'Col_1696', 'ATS2Z', 'Col_229', 'SMR_VSA3', 'Lipinski', 'Col_314', 'SlogP_VSA8', 'Col_1549', 'ATSC2are', 'Col_1089', 'Diameter', 'VSA_EState2', 'GATS1p', 'FCSP3', 'ATSC0i', 'AATSC1i', 'SMR_VSA2', 'piPC5', 'nS', 'Col_561', 'Xc-3dv', 'MATS1p', 'Col_1417', 'AATS1p', 'Col_216', 'Col_798', 'SaasC', 'ATSC3dv', 'PEOE_VSA4', 'PEOE_VSA10', 'NaasC', 'ATSC1pe', 'SsssN', 'NddsN', 'ATS0m', 'ATS1dv', 'PEOE_VSA2', 'Col_739', 'Mi', 'PEOE_VSA1', 'Col_464', 'piPC1', 'EState_VSA8', 'GGI10', 'Col_875', 'Col_950', 'Col_1970', 'Col_385', 'SsCl', 'Col_341', 'ATS6v', 'SssO', 'SlogP_VSA10', 'TopoPSA(NO)', 'Col_608', 'ATSC4d', 'ATSC3i', 'Col_1449', 'Xch-7d', 'Col_714', 'Col_1959', 'Col_495', 'PEOE_VSA13', 'JGI10', 'ATSC2dv', 'SdssC', 'ATSC7i', 'ATSC5dv', 'Col_1905', 'Col_105', 'Col_1034', 'GATS1m', 'Xch-5d', 'Col_1867', 'ZMIC1', 'SlogP_VSA3', 'SMR_VSA5', 'Col_1142', 'Col_2044', 'SMR_VSA4', 'ATS8m', 'Col_162', 'Col_231', 'PEOE_VSA6', 'Col_725', 'Col_248', 'CIC2', 'Mv', 'AATSC0d', 'Col_1911', 'NsssNH', 'GATS1i', 'Xpc-5d', 'Col_716', 'SIC0', 'Col_451', 'Col_1858', 'Col_949', 'nX', 'Col_1655', 'Xc-4d', 'nAromBond', 'C1SP2', 'IC5', 'ATSC2d', 'ATSC2Z', 'Xch-6dv', 'Col_1261', 'ATSC4m', 'Xc-4dv', 'SaasN', 'ATS6dv', 'VSA_EState6', 'ATSC6i', 'Col_611', 'GATS1dv', 'Col_251', 'Col_362', 'VSA_EState3', 'ATSC0p', 'EState_VSA10', 'Col_1917', 'Col_946', 'AATSC0v', 'Col_695', 'Col_797', 'Col_1784', 'Col_980', 'VSA_EState7', 'SlogP_VSA5', 'NaasN', 'ATSC6m', 'Col_381', 'Col_598', 'ATSC3v', 'PEOE_VSA12', 'nBase', 'Col_242', 'Col_423', 'Col_792', 'Col_505', 'JGI2', 'Col_1967', 'Col_494', 'IC2', 'PEOE_VSA5', 'GGI5', 'Col_823', 'Col_638', 'Col_1130', 'Col_184', 'ATSC8are', 'Col_273', 'ATSC3Z', 'Col_1632', 'Col_1771', 'JGI4', 'VSA_EState4', 'AATS1are', 'TSRW10', 'SlogP_VSA11', 'ZMIC4', 'Col_502', 'Col_1639', 'JGI8', 'Col_55', 'VSA_EState9', 'Col_280', 'Col_496', 'ATSC6v', 'Col_456', 'Col_927', 'Col_1347', 'ATSC5v', 'Xch-7dv', 'Col_355', 'ATSC4p', 'Col_1775', 'Col_1635', 'Col_421', 'Col_301', 'Xpc-5dv', 'Col_1693', 'AATS0d', 'ATSC4are', 'Col_1589', 'Col_147', 'Col_1790', 'Col_140', 'MWC07', 'Col_197', 'Col_670', 'MIC0', 'Col_1032', 'RotRatio', 'AATSC1dv', 'ATSC0pe', 'CIC1', 'Col_1070', 'Col_578', 'nBondsD', 'AATSC0dv', 'ATSC2i', 'fMF', 'Col_1827', 'ATSC3d', 'Col_1145', 'ATSC8d', 'JGI7', 'Xch-6d', 'Col_232', 'Col_2029', 'Col_896', 'ATSC8v', 'Col_275', 'JGI5', 'PEOE_VSA11', 'TopoShapeIndex', 'GATS1pe', 'SaaaC', 'Col_563', 'ATSC5d', 'nHRing', 'Col_1171', 'Col_626', 'Col_523', 'C2SP2', 'ATSC1p', 'VSA_EState5', 'Col_1485', 'Col_1734', 'ATSC8Z', 'Col_1412', 'Col_165', 'Mp', 'Col_1984', 'Col_1553', 'SMR_VSA9', 'Col_879', 'ATSC3p', 'StsC', 'AATSC0m', 'Col_557', 'Col_1672', 'Col_278', 'Col_1210', 'Col_94', 'ATSC1d', 'Col_1739', 'Col_1508', 'ATS8dv', 'PEOE_VSA7', 'Xc-6dv', 'Xc-3d', 'Col_559', 'ATSC5Z', 'Col_983', 'Col_224', 'Col_210', 'PEOE_VSA9', 'CIC5', 'Col_1657', 'MIC3', 'JGI6', 'Col_133', 'Col_948', 'Col_214', 'ATSC8i', 'Xch-5dv', 'ATSC1dv', 'Col_1403', 'Col_277', 'Col_318', 'nG12FaHRing', 'EState_VSA3', 'Col_393', 'Col_1656', 'ATSC7pe', 'nSpiro', 'n6HRing', 'Col_1000', 'Col_893', 'ATSC7p', 'n12FARing', 'Col_1845', 'nBondsS', 'ATSC4i', 'Col_265', 'Col_1966', 'Col_1770', 'Col_1537', 'ATSC5i', 'Col_1934', 'GGI2', 'Col_865', 'Col_730', 'Col_268', 'Col_990', 'Col_71', 'Col_776', 'Col_1203', 'EState_VSA2', 'Col_1745', 'ATSC7d', 'Col_1057', 'n4aRing', 'Col_883', 'Col_1573', 'Col_207', 'Col_432', 'Xc-6d', 'Col_1114', 'Col_1385', 'Col_1027', 'ATSC7dv', 'Col_1831', 'Col_503', 'SRW09', 'Col_575', 'AATSC1d', 'ATSC8dv', 'nBondsO', 'Col_1861', 'Col_1388', 'AATSC0are', 'Xc-5d', 'Col_99', 'ATSC5p', 'Col_1169', 'Col_760', 'Col_943', 'n8Ring', 'JGI9', 'SlogP_VSA7', 'Xch-4d', 'Col_1992', 'PEOE_VSA8', 'Col_822', 'Col_1859', 'Col_586', 'Col_1011', 'Col_1601', 'SMR_VSA7', 'Col_1855', 'Col_1713', 'Col_1728', 'Col_2017', 'Col_1814', 'Col_1058', 'Col_127', 'ZMIC2', 'nAromAtom', 'Col_1585', 'ATS5m', 'nHetero', 'Col_1122', 'Col_1355', 'Col_83', 'Col_1168', 'ATSC2v', 'Col_1050', 'C1SP1', 'Col_1188', 'Col_441', 'Col_633', 'Col_1996', 'Col_527', 'Col_1005', 'Col_349', 'Col_173', 'fragCpx', 'Col_962', 'MIC1', 'Col_1157', 'Col_2012', 'Col_978', 'Col_1885', 'ATSC5are', 'nO', 'Col_1759', 'Xc-5dv', 'ATSC6are', 'ATSC7Z', 'Col_84', 'ATS4m', 'Col_842', 'ATSC6p', 'Col_855', 'Col_835', 'ATSC6d', 'Col_960', 'ATSC6dv', 'ATSC4dv', 'n6ARing', 'Col_294', 'AATS1dv', 'Xp-4dv', 'SssssC', 'Col_1152', 'Col_504', 'ATSC3pe', 'Col_1160', 'nFaHRing', 'Col_1866', 'Col_1504', 'Col_809', 'Col_590', 'Col_802', 'Col_935', 'Col_334', 'Col_389', 'Col_1737', 'Col_1164', 'Col_1166', 'Col_350', 'Col_771', 'AATS1Z', 'Col_1364', 'Col_779', 'Col_1061', 'Col_1937', 'Col_651', 'Col_917', 'Col_1304', 'Col_1400', 'Col_738', 'Col_486', 'Col_1580', 'Col_1221', 'Col_1804', 'Col_995', 'Col_388', 'Col_811', 'Col_706', 'Col_1542', 'Col_863', 'Col_2041', 'Col_1311', 'Col_2021', 'Col_1299', 'Col_573', 'Col_580', 'nFRing', 'Col_255', 'AATSC0i', 'Col_996', 'Col_473', 'Col_1240', 'Col_777', 'Col_1738', 'Col_702', 'Col_918', 'Col_839', 'Col_477', 'Col_521', 'Col_1066', 'Col_366', 'nFAHRing', 'Col_506', 'Col_1922', 'Col_1958', 'Col_591', 'Col_614', 'Col_1444', 'Col_476', 'Col_1495', 'Col_1565', 'Col_675', 'Col_1982', 'Col_446', 'Col_1779', 'Col_645', 'Col_1243', 'Col_1281', 'Col_1604', 'Col_544', 'Col_1381', 'Col_1264', 'Col_816', 'Col_1056', 'Col_1852', 'Col_1990', 'Col_1019', 'Col_1295', 'Col_1398', 'Col_1084', 'Col_1613', 'Col_866', 'MZ', 'Col_840', 'NaaO', 'Col_718', 'Col_661', 'Col_1199', 'Col_619', 'Col_2034', 'Col_424', 'Col_1705', 'Col_1547', 'Col_1704', 'Col_1865', 'Col_1889', 'Col_97', 'Col_1266', 'Col_1540', 'Col_376', 'Col_1039', 'Col_1109', 'Col_1920', 'Col_984', 'Col_878', 'Col_1803', 'Col_758', 'Col_1954', 'Col_1554', 'Col_1119', 'Col_852', 'Col_1276', 'Col_459', 'Col_2040', 'Col_750', 'n9FHRing', 'Col_1818', 'Col_815', 'Col_853', 'Col_2026', 'Col_475', 'n10FaRing', 'Col_1481', 'Col_819', 'Col_1624', 'Col_961', 'Col_938', 'Col_1357', 'Col_1012', 'Col_909', 'Col_1459', 'Col_511', 'Col_245', 'Col_1821', 'Col_1570', 'Col_1666', 'Col_1766', 'Col_680', 'Col_1038', 'ATSC0d', 'Col_618', 'Col_1339', 'Col_1471', 'Col_1644', 'Col_1785', 'Col_1810', 'Col_783', 'Col_1721', 'Col_624', 'Col_1891', 'Col_354', 'Col_971', 'Col_1729', 'Col_252', 'Col_1088', 'Col_1174', 'Col_532', 'Col_837', 'Col_1205', 'Col_534', 'Col_1030', 'Col_1177', 'Col_1856', 'Col_1273', 'Col_2032', 'Col_260', 'Col_1329', 'Col_1309', 'Col_1452', 'Col_1136', 'Col_1308', 'Col_1287', 'Col_1087', 'Col_1194', 'Col_1232', 'Col_1929', 'Col_1217', 'Col_1685', 'Col_342', 'Col_327', 'Col_1227', 'Col_378', 'Col_2013', 'Col_1028', 'n5aRing', 'Col_2033', 'Col_807', 'n10FHRing', 'nF', 'Col_1665', 'Col_1753', 'n5ARing', 'Col_1095', 'AATS0dv', 'Col_1230', 'Col_1386', 'Col_1427', 'Col_1825', 'Col_1091', 'Col_1646', 'Col_747', 'Col_1235', 'Col_1252', 'Col_1694', 'Col_1480', 'Col_844', 'Col_1220', 'Col_1106', 'Col_1137', 'Col_920', 'Col_1204', 'Col_1212', 'Col_2028', 'Col_1462', 'Col_2015', 'Col_1489', 'Col_1522', 'Col_1869', 'Col_1453', 'Col_1228', 'Col_1047', 'Col_1008', 'Col_1915', 'Col_1033', 'Col_1393', 'Col_1979', 'Col_841', 'Col_1384', 'Col_833', 'Col_1409', 'Col_1794', 'Col_1455', 'Col_1490', 'Col_1138', 'Col_1768', 'Col_1608', 'Col_1105', 'Col_1463', 'Col_2023', 'Col_1419', 'Col_1121', 'Col_1269', 'Col_941', 'Col_1300', 'Col_1928', 'SdsssP', 'Col_1523', 'Col_1035', 'Col_1260', 'Col_1482', 'Col_474', 'Col_1411', 'n7Ring', 'Col_1413', 'Col_1351', 'Col_1305', 'Col_1706', 'Col_864', 'Col_1757', 'Col_2035', 'Col_1429', 'Col_1762', 'Col_1792', 'Col_1246', 'Col_1637', 'Col_856', 'Col_1795', 'Col_1238', 'Col_1989', 'Col_1956', 'Col_1249', 'Col_1997', 'Col_1839', 'n5HRing', 'Col_1535', 'Col_1620', 'Col_1262', 'Col_1919', 'Col_923', 'Col_1801', 'Col_1185', 'Col_1363', 'Col_1819', 'Col_1747', 'Col_1702', 'Col_1007', 'Col_804', 'Col_1285', 'Col_1173', 'Col_1428', 'Col_1887', 'Col_1835', 'Col_1727', 'Col_1677', 'Col_1510', 'Col_1163', 'Col_836', 'Col_1972', 'Col_1206', 'Col_1090', 'Col_1981', 'Col_736', 'Col_1983', 'Col_1284', 'Col_1591', 'Col_1654', 'Col_1430', 'Col_857', 'Col_1755', 'Col_1722', 'Col_1312', 'Col_1816', 'Col_1291', 'Col_1245', 'Col_1707', 'Col_1914', 'Col_1275', 'Col_1279', 'Col_1296', 'Col_1617', 'Col_1673', 'Col_1606', 'Col_1330', 'Col_1756', 'Col_1310', 'Col_1214', 'Col_1454', 'Col_1501', 'Col_1998', 'Col_1360', 'Col_1496', 'Col_1348', 'Col_1921', 'Col_1631', 'Col_1410', 'Col_1244', 'Col_1720', 'Col_1517', 'Col_1687', 'Col_1754', 'Col_1362', 'Col_1674', 'Col_1551', 'Col_1892', 'Col_1980', 'Col_1382', 'Col_1603', 'Col_1669', 'Col_1248', 'Col_1924', 'Col_1971', 'Col_1609', 'Col_1725', 'Col_2004', 'Col_1733', 'Col_1800', 'Col_1829', 'nG12FRing', 'Col_1947', 'Col_1599', 'Col_1750', 'Col_1499', 'Col_1864', 'Col_1483', 'Col_1986', 'Col_1799', 'Col_1909', 'Col_1479', 'Col_1475', 'Col_1897', 'Col_1520', 'Col_1527', 'Col_1614', 'Col_1995', 'Col_1752', 'Col_1679', 'Col_1650', 'Col_1274', 'Col_1539', 'Col_1850', 'Col_1465', 'Col_1791', 'Col_1683', 'Col_1724', 'Col_1849', 'Col_1991', 'Col_1908', 'Col_1918', 'Col_1847', 'Col_1746', 'Col_1808', 'Col_2003', 'Col_2014', 'Col_2039', 'Col_1815', 'Col_1953', 'Col_2000', 'Col_1773', 'Col_1940', 'Col_1903', 'Col_1778', 'Col_1977', 'Col_1783', 'Col_2042', 'Col_1726', 'Col_1823', 'Col_1708', 'Col_1946', 'Col_2001', 'Col_1964', 'Col_2022', 'Col_2038', 'Col_2007', 'Col_1760', 'Col_1697']
start_column_MM= 255
end_column_MM = 255

def calculate_similarity_threshold(train_features, k, Z=0.5):
    # Calculate pairwise distances between compounds in training set
    pairwise_dist = pairwise_distances(train_features)

    # Calculate k nearest neighbors' distances for each compound
    k_nearest_dist = np.partition(pairwise_dist, k, axis=1)[:, k]

    # Calculate average and standard deviation of k nearest neighbors' distances
    y = np.mean(k_nearest_dist)
    sigma = np.std(k_nearest_dist)

    # Calculate similarity distance threshold
    Dc = Z * sigma + y

    return Dc


# Function to label predictions based on similarity distance threshold
def label_predictions(train_features, test_features, k, Dc):
    # Calculate pairwise distances between compounds in training and test sets
    pairwise_dist = pairwise_distances(train_features, test_features)

    # Find nearest neighbors' distances for each test compound
    nearest_dist = np.partition(pairwise_dist, k-1, axis=0)[k-1, :]

    # Label predictions based on similarity distance threshold
    predictions = np.where(nearest_dist <= Dc, 'Reliable', 'Unreliable')

    return predictions
def analyze_AD(x_train_binary, x_test_binary, x_train_continuous, x_test_continuous, k, Z):
    # Analyze similarity for binary features
    ad = ApplicabilityDomain(verbose=True)
    sims_binary = ad.analyze_similarity(base_test=x_test_binary, base_train=x_train_binary, similarity_metric='tanimoto')
    
    # Calculate similarity threshold for continuous features
    Dc = calculate_similarity_threshold(x_train_continuous, k, Z)
    
    # Get predictions based on continuous features
    predictions_continuous = label_predictions(x_train_continuous, x_test_continuous, k, Dc)
    
    # Combine results and determine ultimate reliability
    sims_binary = sims_binary.reset_index(drop=True)
    df_predictions_continuous = pd.DataFrame({'Prediction': predictions_continuous}).reset_index(drop=True)
    sims_binary['Continous_Reliability'] = df_predictions_continuous['Prediction']
    
    conditions = [
        (sims_binary['Max'] >= 0.8) & (sims_binary['Continous_Reliability'] == 'Reliable'),
        (sims_binary['Max'] >= 0.8) & (sims_binary['Continous_Reliability'] == 'Unreliable'),
        (sims_binary['Max'] < 0.8) & (sims_binary['Continous_Reliability'] == 'Reliable'),
        (sims_binary['Continous_Reliability'] == 'Unreliable')
    ]
    values = ['Absolute reliable', 'Structurally reliable', 'Distance reliable', 'Unreliable']
    sims_binary['Ultimate_reliability'] = np.select(conditions, values)
    
    return sims_binary

def get_applicable_indices(dfs):
    applicable_indices = []
    criteria = ["Structurally reliable", "Distance reliable", "Absolute reliable"]
    for df in dfs:
        applicable_indices.append(set(df[df["Ultimate_reliability"].isin(criteria)].index))
    return applicable_indices



def process_input_for_im(df):
    # Generate Morgan Fingerprints and Mordred Descriptors
    # Load models and scalers lazily
    model_CRU = load_resource_lazy('model_trial_CRUUUZAAAIN.pkl')
    model_LAC = load_resource_lazy('model_For_Lactamase_Without_2_descriptors.pkl')
    model_Sho = load_resource_lazy('Newest_Model_Shoichet&ZINC.pkl')
    scaler_CRU = load_resource_lazy('scaler_Cruzain.pkl')
    scaler_Lac = load_resource_lazy('scaler_Lactamase.pkl')
    scaler_Shoi = load_resource_lazy('scaler_Shoichet.pkl')

    morgan_fingerprints = morgan_fpts(df['SMILES'])
    mordred_descriptors = All_Mordred_descriptors(df['SMILES'])
    
    # Combine Morgan Fingerprints and Mordred Descriptors
    Morgan_fingerprints_df = pd.DataFrame(morgan_fingerprints, columns=['Col_{}'.format(i) for i in range(morgan_fingerprints.shape[1])])
    exclude_cols = ['Agg status']
    cols_to_add = [col for col in mordred_descriptors.columns if col not in exclude_cols]
    Morgan_fingerprints_df = pd.concat([Morgan_fingerprints_df, mordred_descriptors[cols_to_add]], axis=1)
    cat_cols = Morgan_fingerprints_df.select_dtypes(include=['object']).columns.tolist()
    Morgan_fingerprints_df = Morgan_fingerprints_df.drop(cat_cols, axis=1)
    results = {}
    
    results = {
        'Cruzain': process_target(Morgan_fingerprints_df, 'Cruzain', descriptor_columns_cru, preprocess_and_scale_Cru, model_CRU, start_column_Cru, end_column_Cru, 'Cruzain', scaler_CRU).to_dict('records'),
        'Lactamase': process_target(Morgan_fingerprints_df, 'Lactamase', descriptor_columns_Lac, preprocess_and_scale_Lac, model_LAC, start_column_Lac, end_column_Lac, 'Lactamase', scaler_Lac).to_dict('records'),
        'Shoichet': process_target(Morgan_fingerprints_df, 'Shoichet', descriptor_columns_Sho, preprocess_and_scale_Sho, model_Sho, start_column_Shoi, end_column_Shoi, 'Shoichet', scaler_Shoi).to_dict('records')
    }
    
    return results

def process_input_for_cm(df):
    # Generate Morgan Fingerprints and Mordred Descriptors
    model_CRU = load_resource_lazy('model_trial_CRUUUZAAAIN.pkl')
    model_LAC = load_resource_lazy('model_For_Lactamase_Without_2_descriptors.pkl')
    model_Sho = load_resource_lazy('Newest_Model_Shoichet&ZINC.pkl')
    scaler_CRU = load_resource_lazy('scaler_Cruzain.pkl')
    scaler_Lac = load_resource_lazy('scaler_Lactamase.pkl')
    scaler_Shoi = load_resource_lazy('scaler_Shoichet.pkl')

    morgan_fingerprints = morgan_fpts(df['SMILES'])
    mordred_descriptors = All_Mordred_descriptors(df['SMILES'])

    # Combine Morgan Fingerprints and Mordred Descriptors
    Morgan_fingerprints_df = pd.DataFrame(morgan_fingerprints, columns=['Col_{}'.format(i) for i in range(morgan_fingerprints.shape[1])])
    exclude_cols = ['Agg status']
    cols_to_add = [col for col in mordred_descriptors.columns if col not in exclude_cols]
    Morgan_fingerprints_df = pd.concat([Morgan_fingerprints_df, mordred_descriptors[cols_to_add]], axis=1)
    cat_cols = Morgan_fingerprints_df.select_dtypes(include=['object']).columns.tolist()
    Morgan_fingerprints_df = Morgan_fingerprints_df.drop(cat_cols, axis=1)

    # Prepare data and function arguments for process_target_for_cm
    descriptor_columns = [descriptor_columns_cru, descriptor_columns_Lac, descriptor_columns_Sho]
    preprocess_and_scale_functions = [preprocess_and_scale_Cru, preprocess_and_scale_Lac, preprocess_and_scale_Sho]
    models = [model_CRU, model_LAC, model_Sho]
    start_columns = [start_column_Cru, start_column_Lac, start_column_Shoi]
    end_columns = [end_column_Cru, end_column_Lac, end_column_Shoi]
    target_prefixes = ['Cruzain', 'Lactamase', 'Shoichet']
    scalers = [scaler_CRU, scaler_Lac, scaler_Shoi]

    # Get consensus results
    consensus_results = process_target_for_cm(Morgan_fingerprints_df, descriptor_columns, preprocess_and_scale_functions, models, start_columns, end_columns, target_prefixes, scalers)

    return consensus_results

def process_input_for_mm(df):
    # Generate Morgan Fingerprints and Mordred Descriptors
    # Load models and scalers lazily
    model_MM = load_resource_lazy('MM_model.pkl')
    scaler_MM = load_resource_lazy('scaler_MM.pkl')

    # Generate Morgan Fingerprints and Mordred Descriptors
    morgan_fingerprints = morgan_fpts(df['SMILES'])
    mordred_descriptors = All_Mordred_descriptors(df['SMILES'])

    # Combine Morgan Fingerprints and Mordred Descriptors
    Morgan_fingerprints_df = pd.DataFrame(morgan_fingerprints, columns=['Col_{}'.format(i) for i in range(morgan_fingerprints.shape[1])])
    exclude_cols = ['Agg status']
    cols_to_add = [col for col in mordred_descriptors.columns if col not in exclude_cols]
    Morgan_fingerprints_df = pd.concat([Morgan_fingerprints_df, mordred_descriptors[cols_to_add]], axis=1)
    cat_cols = Morgan_fingerprints_df.select_dtypes(include=['object']).columns.tolist()
    Morgan_fingerprints_df = Morgan_fingerprints_df.drop(cat_cols, axis=1)

    # Process for MM target
    results = {
        'MM': process_target(
            Morgan_fingerprints_df, 
            'MM', 
            descriptor_columns_MM, 
            preprocess_and_scale_MM, 
            model_MM, 
            start_column_MM, 
            end_column_MM, 
            'MM', 
            scaler_MM
        ).to_dict('records')
    }
    
    return results



def process_target(Morgan_fingerprints_df, target_name, descriptor_columns, preprocess_and_scale, model, start_column, end_column, target_prefix, scaler):
    X_test = Morgan_fingerprints_df[descriptor_columns]
    X_test_scaled = preprocess_and_scale(X_test, scaler)
    
    # Load Binary features
    with open(f'x_train1_{target_prefix}_Binary_training_set.pkl', 'rb') as f:
        x_train1_b_array = pickle.load(f)
    
    # Load Continuous features
    with open(f'x_train1_{target_prefix}_Continuous_training_set.pkl', 'rb') as f:
        x_train1_c_array = pickle.load(f)
    
    # Binary features
    x_test2_b_array = X_test_scaled[:, start_column:]
    
    # Continuous features
    x_test2_c_array = X_test_scaled[:, :end_column]
    
    # AD analysis and predictions
    sims = analyze_AD(x_train1_b_array, x_test2_b_array, x_train1_c_array, x_test2_c_array, k=5, Z=0.5)
    if 'Ultimate_reliability' not in sims.columns:
        raise ValueError("Ultimate_reliability column is missing in AD analysis results")

    # Get indices of applicable molecules
    applicable_indices = get_applicable_indices([sims])
    
    # Free up memory
    del x_train1_b_array, x_train1_c_array
    
    predictions_proba = model.predict_proba(X_test_scaled)[:, 1]
    results = []    
    for i, prob in enumerate(predictions_proba):
        if i in applicable_indices[0]:
            domain_status = "Inside the domain"
        else:
            domain_status = "Outside the domain"
    
        if prob > 0.6:
            prediction = '1 (High Confidence)'
        elif prob < 0.3:
            prediction = '0 (High Confidence)'
        else:
            prediction = 'Ambiguous'
    
    results_df = pd.DataFrame(predictions_proba, columns=['Prediction Probability'])
    results_df['Domain Status'] = ['Inside the domain' if i in applicable_indices[0] else 'Outside the domain' for i in range(len(predictions_proba))]
    results_df['Prediction'] = ['1 (High Confidence)' if prob > 0.6 else '0 (High Confidence)' if prob < 0.3 else 'Ambiguous' for prob in predictions_proba]
    results_df['Ultimate_reliability'] = sims['Ultimate_reliability']  # Add the Ultimate_reliability column

    return results_df

def process_target_for_cm(Morgan_fingerprints_df, descriptor_columns, preprocess_and_scale_functions, models, start_columns, end_columns, target_prefixes, scalers):
    # Prepare a list to store results from each model
    all_model_results = []

    # Process each model
    for target_name, descriptor_columns, preprocess_and_scale, model, start_column, end_column, target_prefix, scaler in zip(
            target_prefixes, descriptor_columns, preprocess_and_scale_functions, models, start_columns, end_columns, target_prefixes, scalers):

        X_test = Morgan_fingerprints_df[descriptor_columns]
        X_test_scaled = preprocess_and_scale(X_test, scaler)

        # Load binary and continuous features from training set
        with open(f'x_train1_{target_prefix}_Binary_training_set.pkl', 'rb') as f:
            x_train1_b_array = pickle.load(f)
        with open(f'x_train1_{target_prefix}_Continuous_training_set.pkl', 'rb') as f:
            x_train1_c_array = pickle.load(f)

        # Prepare binary and continuous features for test set
        x_test2_b_array = X_test_scaled[:, start_column:]
        x_test2_c_array = X_test_scaled[:, :end_column]

        # AD analysis
        sims = analyze_AD(x_train1_b_array, x_test2_b_array, x_train1_c_array, x_test2_c_array, k=5, Z=0.5)
        if 'Ultimate_reliability' not in sims.columns:
            raise ValueError("Ultimate_reliability column is missing in AD analysis results")

        # Predictions
        predictions_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Store results
        model_results = {
            'Model Name': target_name,  # Assuming target_name is your model identifier
            'Probability': predictions_proba,
            'Domain Status': ['Inside domain' if i in get_applicable_indices([sims])[0] else 'Outside domain' for i in range(len(predictions_proba))],
        }
        all_model_results.append(model_results)

    # Prepare consensus results
    consensus_results = []
    for i in range(len(Morgan_fingerprints_df)):
        inside_domains = [(model_results['Model Name'], model_results['Probability'][i]) for model_results in all_model_results if model_results['Domain Status'][i] == 'Inside domain']
        domain_count = len(inside_domains)

        if domain_count > 0:
            avg_prob = np.mean([prob for _, prob in inside_domains])
            domain_names = ', '.join([name for name, _ in inside_domains])
            prediction_status = f'Inside {domain_count} domains: {domain_names}'
        else:
            avg_prob = np.mean([model_results['Probability'][i] for model_results in all_model_results])
            prediction_status = 'Outside all domains'

        prediction = '1 (High Confidence)' if avg_prob > 0.6 else '0 (High Confidence)' if avg_prob < 0.3 else 'Ambiguous'

        consensus_results.append({'Compound Index': i, 'Consensus Prediction': prediction, 'Status': prediction_status})

    return consensus_results


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('BAD Molecule Filter.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_type = data.get('model_type', 'IM')  # Default to IM if not specified
    smiles = data['smiles']
    df = pd.DataFrame({'SMILES': [smiles]})
    
    if model_type == 'IM':
        results = process_input_for_im(df)
    elif model_type == 'CM':
        results = process_input_for_cm(df)
    elif model_type == 'MM':
        results = process_input_for_mm(df)  # Handling Merged Model
    else:
        return jsonify({'error': 'Invalid model type specified'})
    
    results_df = pd.DataFrame.from_dict(results)

    results_html = results_df.to_html(classes='table table-striped', index=False, border=0)

    return results_html

@app.route('/upload_file', methods=['POST'])
def upload_file():
    model_type = request.form.get('model_type', 'IM')  # Default to IM if not specified

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Additional checks for file size, type, etc. can be added here

    if file:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file type'})
        
        if model_type == 'IM':
            results = process_input_for_im(df)
        elif model_type == 'CM':
            results = process_input_for_cm(df)
        elif model_type == 'MM':
            results = process_input_for_mm(df)  # Handling Merged Model
        else:
            return jsonify({'error': 'Invalid model type specified'})

        results_df = pd.DataFrame(results)  # Directly create DataFrame from the list

        results_html = results_df.to_html(classes='table table-striped', index=False, border=0)

        return results_html

if __name__ == '__main__':
    app.run(debug=True)