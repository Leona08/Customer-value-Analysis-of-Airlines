import numpy as np
import pandas as pd

# 缺失值处理
print('原始数据的形状为：', data.shape)

# 去掉票价为空的记录
airline_notnull = data.loc[data['SUM_YR_1'].notnull()&data['SUM_YR_2'].notnull(),:]
print('删除缺失记录后数据的大小为：', airline_notnull.shape)

# 只保留票价非零的，或者平均折扣率不为0且总飞行公里数大于0的记录
index1 = airline_notnull['SUM_YR_1'] != 0
index2 = airline_notnull['SUM_YR_2'] != 0
index3 = (airline_notnull['SEG_KM_SUM'] > 0)&(airline_notnull['avg_discount'] != 0)
index4 = airline_notnull['AGE'] > 100
airline = airline_notnull[(index1 | index2) & index3 & ~index4]

print('数据清洗后数据的大小为：', airline.shape)

# 属性选择
airline_selection = airline[['FFP_DATE','LOAD_TIME','LAST_TO_END',
                             'FLIGHT_COUNT','SEG_KM_SUM','avg_discount']]
print('筛选的属性前5行为：\n', airline_selection.head(5))

# 构造属性L
L = pd.to_datetime(airline_selection['LOAD_TIME']) - pd.to_datetime(airline_selection['FFP_DATE'])
L = L.astype('str').str.split().str[0]
L = L.astype('int')/30
# 合并属性
airline_features = pd.concat([L, airline_selection.iloc[:,2:]], axis = 1)
print('构建的LRFMC属性前5行为：\n', airline_features.head(5))

# 数据标准化
from sklearn.preprocessing import StandardScaler
data = StandardScaler().fit_transform(airline_features)
print('标准化后LRFMC 5个属性为：\n', data[:5,:])