import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

'''
客户基本信息：
MEMBER_NO 会员卡号
FFP_DATE 入会时间
FIRST_FLIGHT_DATE 第一次飞行日期
GENDER 性别
FFP_TIER 会员卡级别
WORK_CITY 工作地城市
WORK_PROVINCE 工作地所在省份
WORK_COUNTRY 工作地所在国家
AGE 年龄

乘机信息：
LOAD_TIME 统计的结束时间
FLIGHT_COUNT 统计的飞行次数
LAST_TO_END 最后一次乘机时 统计结束的时长
avg_discount 平均折扣系数
SUN_YR1/2 统计第1/2年的票价收入
SEG_KM_SUM 统计的总飞行千米数
LAST_FLIGHT_DATE 末次飞行日期
AVG_INTERVAL 平均乘机的时间间隔
MAX_INTERVAL 最大乘机间隔

积分信息：
EXCHANGE_COUNT 积分兑换次数
EP_SUM 总精英积分
BP_SUM 总基本积分
'''
#1.读取air文件，编码格式gb18030 使用pandas读取这个文件
airline_data = pd.read_csv('./data/air_data.csv', encoding='gb18030')
#print(airline_data.shape)

#去除票价为空的记录
#查看每一列是否有空值
exp1 = airline_data['SUM_YR_1'].notnull()
#print(exp1.shape)
exp2 = airline_data['SUM_YR_2'].notnull()
#print(exp2.shape)
#同时为True才是True
exp = exp1 & exp2
print(type(exp))

#exp表示的是(exp1&exp2)值为True的所有数据
airline_data_notnull = airline_data.loc[exp, :]
print(airline_data_notnull.shape)

#筛选数据：只保留票价非零，或者平均折扣不为零的且总飞行千米数大于零的记录
index1 = airline_data_notnull['SUM_YR_1'] != 0
index2 = airline_data_notnull['SUM_YR_2'] != 0
index3 = (airline_data_notnull['SEG_KM_SUM'] > 0) & (airline_data_notnull['avg_discount'] != 0)

airline = airline_data_notnull[(index1|index2) & index3]
print(airline.shape)

#在统计数据上，针对航空公司的客户，有一个LRFMC模型
#处理之后的一种特征

#选取需要的特征
airLine_selection = airline[['FFP_DATE', 'LOAD_TIME', 'FLIGHT_COUNT', 'LAST_TO_END', 'avg_discount', 'SEG_KM_SUM']]

#构建L特征值
L = pd.to_datetime(airLine_selection['LOAD_TIME']) - pd.to_datetime(airLine_selection['FFP_DATE'])

#L = L.astype('str').str
L = L.astype("str").str.split( ).str[0]
L = L.astype("int") / 30

#合并特征
airLine_features = pd.concat([L, airLine_selection.iloc[:, 2:]], axis = 1)
print(airLine_features.head())

#数据标准化处理
from sklearn.preprocessing import StandardScaler
#fit_transform把生成规则和应用规则合在一起
data = StandardScaler().fit_transform(airLine_features)

#k-means算法进行客户的据类
#构建模型 分成3， 4， 6， 7类都可以
kmeans_model = KMeans(n_clusters = 5, random_state = 123)
kmeans_model.fit(data)
#打印中心点
print(kmeans_model.cluster_centers_)
#打印标签
print(kmeans_model.labels_)
print(pd.Series(kmeans_model.labels_).value_counts())

#【项目一】航空公司价值分析
# 自己写注释 （必写，每一行代码解释）（查重相似度95%以上，直接60）C类
# 自己做数据分析，画3个以上的图，并对图的意义进行说明 B类
# PAC降维之后可视化  B类

pca_model = PCA(n_components=2).fit(data)
pca_data = pca_model.transform(data)

df = pd.DataFrame(pca_data)
df['labels'] = kmeans_model.labels_
df1 = df[df['labels'] == 0]
df2 = df[df['labels'] == 1]
fig = plt.figure(figsize=(9, 6))

plt.plot(df1[0], df1[1], 'bo',
         df2[0], df2[1], 'r*')
plt.show()
