# 构建k-means模型
from sklearn.cluster import KMeans
k = 5
# 构建模型，随机种子设为123
kmeans_model = KMeans(n_clusters = k, n_jobs = 4, random_state = 123)
fit_kmeans = kmeans_model.fit(data)
# 查看聚类结果
kmeans_cc = kmeans_model.cluster_centers_     # 聚类中心
print('各类聚类中心为：\n', kmeans_cc)
kmeans_labels = kmeans_model.labels_       # 样本类别的标签

print('各样本的类别标签为：\n',kmeans_labels)

r1 = pd.Series(kmeans_model.labels_).value_counts()
print('最终每个类别的数目为：\n', r1)
# 输出聚类分群的结果
cluster_center = pd.DataFrame(kmeans_model.cluster_centers_, 
                              columns = ['ZL','ZR','ZF','ZM','ZC'])
cluster_center.index = pd.DataFrame(kmeans_model.labels_).drop_duplicates().iloc[:,0]

print(cluster_center)

# 2. 客户价值分析

import matplotlib.pyplot as plt
labels = ['ZL','ZR','ZF','ZM','ZC']
legen = ['客户群' + str(i+1) for i in cluster_center.index]
lstype = ['-','--',(0,(3,5,1,5,1,5)),':','-.']
kinds = list(cluster_center.iloc[:,0])  # 取了ZL列
print(kinds)
print(cluster_center.head(5))
print(cluster_center.index)
print(legen)

# 由于雷达图要保证数据闭合，因此在添加L列，并转换为np.ndarray
cluster_center = pd.concat([cluster_center, cluster_center[['ZL']]], axis=1)
# cluster_center.iloc[:,0:6]
centers = np.array(cluster_center.iloc[:,0:6])

# 分割圆周长，让其闭合
n = len(labels)
angle = np.linspace(0, 2 * np.pi, n, endpoint = False)
angle = np.concatenate((angle, [angle[0]]))
print(angle)
print(centers)

# 绘图
fig = plt.figure(figsize = (8, 6))
ax = fig.add_subplot(111, polar = True)
for i in range(len(kinds)):
    ax.plot(angle, centers[i], linestyle=lstype[i], linewidth = 2, label=kinds[i])
# 添加属性标签
ax.set_thetagrids(angle * 180 / np.pi, labels)
plt.title('客户特征分析雷达图')
plt.legend(legen)
plt.show()