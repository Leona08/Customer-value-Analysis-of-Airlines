import numpy as np 
import pandas as pd 

data = pd.read_csv('air_data.csv')

# 数据预处理
print('原始数据的形状为：', data.shape)

# 去掉票价为空的记录
airline_notnull = data.loc[data['SUM_YR_1'].notnull()&data['SUM_YR_2'].notnull(),:]

# 只保留票价非零的，或者平均折扣率不为0且总飞行公里数大于0的记录
index1 = airline_notnull['SUM_YR_1'] != 0
index2 = airline_notnull['SUM_YR_2'] != 0
index3 = (airline_notnull['SEG_KM_SUM'] > 0)&(airline_notnull['avg_discount'] != 0)
index4 = airline_notnull['AGE'] > 100
airline = airline_notnull[(index1 | index2) & index3 & ~index4]

print('数据清洗后数据的大小为：', airline.shape)

# 属性选择
data_attr = airline[['FFP_TIER','AVG_INTERVAL','avg_discount','BP_SUM',
                     'WEIGHTED_SEG_KM','EXCHANGE_COUNT','Points_Sum','Point_NotFlight',
                    'Eli_Add_Point_Sum','FLIGHT_COUNT','SEG_KM_SUM','EP_SUM','ADD_Point_SUM',
                    'P1Y_Flight_Count', 'L1Y_Flight_Count']]
# 观察一下数据的缺失值情况
explore = data_attr.describe(percentiles=[],include='all').T
explore['null'] = len(data_attr) - explore['count']
explore = explore[['null','max','min']]
explore.columns = [u'空数值',u'最大值',u'最小值']

# 将飞行次数超过6次的客户记为 老客户 重点关注这部分客户的流失情况
index1 = data_attr['FLIGHT_COUNT'] > 6
data_feature = data_attr[index1]

# 添加标签列 第二年飞行是第二年飞行次数90% 正常客户；低于50% 记为流失客户 
test2 = data_feature
test2['class'] = test2['L1Y_Flight_Count'] / test2['P1Y_Flight_Count']

# 更改class这一列的值
def change_class(x):
    if x > 0.5:
        return(0)
    else:
        return(1)
test2['class'] = test2['class'].apply(change_class)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, scale
# X 值的提取

train_data = test2.iloc[1:,0:13]
class_data = test2.iloc[1:, -1]
tmp = np.array(train_data.iloc[:,0:1])
tmp2 = OneHotEncoder(sparse=False).fit_transform(tmp[:, (0,)])
X = np.hstack((train_data.iloc[:,1:], tmp2))
y = np.array(class_data)
X = scale(X)

# 训练集和测试集划分
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 123)

# 构建模型
clf = RandomForestClassifier()
clf.fit(train_x, train_y)
predict_y = clf.predict(test_x)
acc = clf.score(test_x, test_y)
print(acc)

# 混淆矩阵的计算
from sklearn.metrics import confusion_matrix, precision_recall_curve
cm = confusion_matrix(test_y, predict_y)
print(cm)

# 混淆矩阵可视化

def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix"', cmap = plt.cm.Blues) :
    plt.figure()
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)
 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
        plt.text(j, i, cm[i, j],
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
class_names = [0,1]
plot_confusion_matrix(cm, classes=class_names,title = 'RFC Confusion Matrix')

# 显示模型评估结果
def show_metrics():
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    tn = cm[0,0]
    print('精确率: {:.3f}'.format(tp/(tp+fp)))
    print('召回率: {:.3f}'.format(tp/(tp+fn)))
    print('F1值: {:.3f}'.format(2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn))))))
# 绘制精确率-召回率曲线
def plot_precision_recall():
    plt.step(recall, precision, color = 'b', alpha = 0.2, where = 'post')
    plt.fill_between(recall, precision, step ='post', alpha = 0.2, color = 'b')
    plt.plot(recall, precision, linewidth=2)
    plt.xlim([0.0,1])
    plt.ylim([0.0,1.05])
    plt.xlabel('Recall Rate')
    plt.ylabel('Precision Rate')
    plt.title('Recall-Precision Curve')
    plt.show()

score_y = clf.predict_proba(test_x)[:,1]
precision, recall, thresholds = precision_recall_curve(test_y, score_y)
plot_precision_recall()

# 计算特征重要性
clf.feature_importances_

# 绘制特征重要性直方图
import matplotlib.pyplot as plt
Feature_importances = [0.14624663, 0.13801399, 0.12229524, 0.1288424 , 0.01804451,
       0.12343703, 0.03250733, 0.03173134, 0.07327083, 0.12983688,
       0.01387499, 0.02876411, 0.00789754, 0.00377183, 0.00146534]
 
fea_label = ['FFP_TIER_1','FFP_TIER_2','FFP_TIER_3', 'AVG_INTERVAL', 'avg_discount', 'BP_SUM', 'WEIGHTED_SEG_KM',
       'EXCHANGE_COUNT', 'Points_Sum', 'Point_NotFlight', 'Eli_Add_Point_Sum',
       'FLIGHT_COUNT', 'SEG_KM_SUM', 'EP_SUM', 'ADD_Point_SUM']
Feature_importances = [round(x,4) for x in Feature_importances]
F2 = pd.Series(Feature_importances,index = fea_label)
F2 = F2.sort_values(ascending = True)
f_index = F2.index
f_values = F2.values
 
# -*-输出 -*- # 
print ('f_index:',f_index)
print ('f_values:',f_values)
#####################################
x_index = list(range(0,15))
x_index = [x/15 for x in x_index]
plt.rcParams['figure.figsize'] = (10,10)
plt.barh(x_index,f_values,height = 0.028 ,align="center",color = 'orange',tick_label=f_index)
plt.xlabel('特征重要性')
plt.ylabel('特征')
plt.show()