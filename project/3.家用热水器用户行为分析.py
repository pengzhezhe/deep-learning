# 家用热水器用户行为分析与事件识别
# 数据抽取：历史数据（选择性抽取）、增量数据（实时监控）
# 数据预处理：数据清洗、数据标准化、特征构建
# 分析与建模：神经网络
import pandas as pd
import numpy as np

data=pd.read_excel("./data/original_data.xls")
# print(data.shape)
data.drop(labels=['热水器编号','有无水流','节能模式'],axis=1,inplace=True)
data.to_csv("./tmp/water_heater.csv",index=False)

# 划分用水事件：确定单次用水时长间隔、计算两条相隔记录的时间
threshold=pd.Timedelta('4 min')# 设置阈值
data['发生时间'] = pd.to_datetime(data['发生时间'],format="%Y%m%d%H%M%S") # 转换时间格式
data =data[data['水流量']>0] # 只要水流量大于0的记录
# 相邻时间向前差分，比较是否大于阈值
sjKs = data['发生时间'].diff()>threshold
sjKs.iloc[0]=True # 令第一个时间为第一个用水时间的开始事件
# 向后差分的结果
sjJs=sjKs[1:]
# 将最后一个时间作为最后一个用水事件的结束事件
sjJs= pd.concat([sjJs,pd.Series(True)])
# print(sjJs)
# 创建数据框，并定义用水事件的序列
sj = pd.DataFrame(np.arange(1,sum(sjKs)+1),columns=['事件序号'])
sj['事件起始编号']= data.index[sjKs==1]+1
sj['事件的终止编号']=data.index[sjJs==1]+1
print("用水时间超过4min的事件",sj.shape[0])
# 【项目三】基于python对热水器用户数据采集进行数据挖掘 B类
#  挖掘3个以上的有意义的数据，并画图，就完成任务。
#  如果使用了神经网络算法，就算A类