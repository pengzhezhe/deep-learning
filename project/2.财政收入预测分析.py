import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

"""
x1	社会从业人数
x2	在岗职工工资总额
x3	社会消费品零售总额
x4	城镇居民人均可支配收入
x5	城镇居民人均消费性支出
x6	年末总人口
x7	全社会固定资产投资额
x8	地区生产总值
x9	第一产业产值
x10	税收
x11 居民消费价格指数
x12 第三产业与第二产业产值比
x13 居民消费水品
"""
data = pd.read_csv("./data/data.csv", encoding="utf8", sep=",")

# 各个特征值feature的相关性
print(data.corr(method="pearson"))

# 线性回归拟合也可
# Lasso回归：回归方法的一种，利用惩正则化方法
lasso = Lasso(1000, random_state=1234)

lasso.fit(data.iloc[:, 0:13], data['y'])
# 相关系数
# print(lasso.coef_)

print(np.sum(lasso.coef_ != 0))
mask = lasso.coef_ != 0
new_data = data.iloc[:, mask]
# print(new_data)

# 此时Lasso模型已经训练好了，可以预测数据
# 也可用有效的数据在训练一次
# 【项目二】（全部完成就算B类）
# B类：可以用线性回归，也能用Lasso回归
# 把预测值和真实值比较的图画出来
# 自己做数据分析，画3个以上的图，并对图的意义进行说明
pre = lasso.predict(data.iloc[0:1, 0:13])
print("预测值：",pre)
print(data.iloc[0:1, 13])
