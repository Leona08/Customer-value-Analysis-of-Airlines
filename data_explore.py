import pandas as pd
datafile = 'air_data.csv'
data = pd.read_csv(datafile, encoding = 'utf-8')
# 查看字段个数和字段名
print(data.columns)
print(len(list(data.columns)))

# 对数据的基本描述，percentiles参数是指定计算多少的分位数表
explore = data.describe(percentiles=[],include='all').T

# describe()函数自动计算非空值数，需要手动计算空值数
explore['null'] = len(data) - explore['count']
explore = explore[['null','max','min']]
explore.columns = [u'空数值',u'最大值',u'最小值']  # 表头重新命名

# 分布分析
# 客户信息类别
# 提取会员入会年份
from datetime import datetime
ffp = data['FFP_DATE'].apply(lambda x:datetime.strptime(x,'%Y/%m/%d'))
ffp_year = ffp.map(lambda x:x.year)

# 绘制各年份会员入会人数直方图
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (8,5))

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.hist(ffp_year, bins='auto', color='#0504aa')
plt.xlabel('年份')
plt.ylabel('入会人数')
plt.title('各年份入会人数')
plt.show()

# 提取不同性别人数
male = pd.value_counts(data['GENDER'])['男']
female = pd.value_counts(data['GENDER'])['女']
# 绘制会员性别比例饼图
fig = plt.figure(figsize=(7,4))
plt.pie([male, female], labels=['男','女'], colors = ['lightskyblue','lightcoral'], autopct = '%1.1f%%')
plt.title('会员性别比例')
plt.show()

# 提取不同级别会员的人数
lv_four = pd.value_counts(data['FFP_TIER'])[4]
lv_five = pd.value_counts(data['FFP_TIER'])[5]
lv_six = pd.value_counts(data['FFP_TIER'])[6]
# 绘制会员各级别人数条形图
fig = plt.figure(figsize=(8,5))
plt.bar(x=range(3),height=[lv_four, lv_five, lv_six], width=0.4, alpha=0.8, color='darkblue')
plt.xticks([index for index in range(3)],['4','5','6'])
plt.xlabel('会员等级')
plt.ylabel('会员人数')
plt.title('会员各级人数')
plt.show()

# 提取会员年龄
age = data['AGE'].dropna()
age = age.astype('int64')
# 绘制会员年龄分布箱线图
fig = plt.figure(figsize=(5,10))
plt.boxplot(age,
           patch_artist = True, # 上下四分位框是否填充
           labels = ['会员年龄'],
           boxprops = {'facecolor':'lightblue'}) # 设置填充颜色
plt.title('会员年龄分布箱线图')
plt.grid(axis = 'y')
plt.show()

# 客户乘机信息分布分析
#选取最后一次乘机至结束的时长，客户乘机信息中的飞行次数，总飞行公里数进行探索分析。
lte = data['LAST_TO_END']
fc = data['FLIGHT_COUNT']
sks = data['SEG_KM_SUM'] # 观测窗口总飞行公里数
# 绘制最后乘机至结束时长箱型图
fig = plt.figure(figsize = (5,8))
plt.boxplot(lte, patch_artist=True, labels=['时长'],boxprops = {'facecolor':'lightblue'})
plt.title('会员最后乘机至结束时长分布箱型图')
plt.grid(axis = 'y')
plt.show()

# 绘制客户飞行次数箱型图
fig = plt.figure(figsize = (5,8))
plt.boxplot(fc, patch_artist=True, labels=['飞行次数'],boxprops = {'facecolor':'lightblue'})
plt.title('会员飞行次数分布箱型图')
plt.grid(axis = 'y')
plt.show()

# 绘制客户总飞行公里数箱线图
fig = plt.figure(figsize = (5,8))
plt.boxplot(sks, patch_artist=True, labels=['总飞行公里数'],boxprops = {'facecolor':'lightblue'})
plt.title('客户总飞行公里数箱线图')
plt.grid(axis = 'y')
plt.show()

# 积分信息类别
# 提取会员积分兑换次数
ec = data['EXCHANGE_COUNT']
# 绘制会员兑换次数直方图
fig = plt.figure(figsize=(8,5))
plt.hist(ec, bins = 10, color = '#0504aa')
plt.xlabel('兑换次数')
plt.ylabel('会员人数')
plt.title('会员兑换积分次数分布直方图')
plt.show()

# 提取会员总累计积分
ps = data['Points_Sum']
# 提取会员总累计积分箱型图
fig = plt.figure(figsize = (5,8))
plt.boxplot(ps, patch_artist = True, labels = ['总累计积分'],boxprops = {'facecolor':'lightblue'})
plt.title('客户总累计积分箱型图')
plt.grid(axis = 'y')
plt.show()

# 相关性分析
# 提取属性合并为新数据集
data_corr = data[['FFP_TIER', 'FLIGHT_COUNT','LAST_TO_END','SEG_KM_SUM','EXCHANGE_COUNT','Points_Sum']
                ]
age1 = data['AGE'].fillna(0)
data_corr['AGE'] = age1.astype('int64')
data_corr['ffp_year'] = ffp_year
# 计算相关性矩阵
dt_corr = data_corr.corr(method = 'pearson')
print('相关性矩阵为：\n', dt_corr)
# 绘制热力图
import seaborn as sns
plt.subplots(figsize = (10,10))
sns.heatmap(dt_corr, annot=True, vmax = 1, square = True)
plt.show()