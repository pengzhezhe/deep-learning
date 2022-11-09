import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载iris数据集
iris = load_iris()
# 数据的特征值
iris_data = iris['data']
# 数据标签值 0 1 2
iris_target = iris['target']

# 处理数据的标签值
# 0-> [1,0,0]
# 1-> [0,1,0]
# 2-> [0,0,1]
iris_label = np.zeros((iris_target.shape[0], 3))
for i in range(iris_target.shape[0]):
    iris_label[i, iris_target[i]] = 1

# 切割训练集和测试集，训练集80% 测试集20%
iris_data_train, iris_data_test, iris_label_train, iris_label_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=123)
# 标准化数据
# 生成标准化规则
stdScaler = StandardScaler().fit(iris_data_train)
# 应用规则到训练集、测试集
iris_data_trainStd = stdScaler.transform(iris_data_train)
iris_data_testStd = stdScaler.transform(iris_data_test)

# 数据与标签的占位
# 输入值，IRIS输入层有4个神经元，None表示输入样本的数量暂不确定，可输入多个样本
x = tf.placeholder(tf.float32, shape=[None, 4])
# 输出值，IRIS将数据分为3类，输出层有3个神经元，None表示输入样本的数量暂不确定
# 真实值，后面用于和预测值比较计算准确率
y_actual = tf.placeholder(tf.float32, shape=[None, 3])

# 初始化权重和偏置，后面训练时需要更新
W = tf.Variable(tf.zeros([4, 3]))
b = tf.Variable(tf.zeros([3]))

# 由于iris分类问题是一个多分类问题，所以激活函数选用softmax得到预测值
y_predict = tf.nn.softmax(tf.matmul(x, W) + b)

# 预测值和真实值，通过交叉熵函数，得到损失函数
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_predict, labels=tf.argmax(y_actual, 1))

# 通过梯度下降算法使得残差最小
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 计算准确率
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
# 多个批次准确率均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# 初始化变量 tf.zeros权重和偏置初始化为0
init = tf.global_variables_initializer()
# 打开会话
with tf.Session() as sess:
    # 初始化变量
    sess.run(init)
    # 训练轮数为训练集样本数除以10，一轮喂10个样本
    for i in range(iris_label_train.shape[0] // 10):
        # 训练集批量喂入，一次喂10个样本
        data = iris_data_trainStd[i * 10:i * 10 + 10, :]
        label = iris_label_train[i * 10:i * 10 + 10, :]
        # 执行梯度下降算法，每执行一次，更新一次权重和偏置
        sess.run(train_step, feed_dict={x: data, y_actual: label})
        # 每更新一次参数，就使用测试集计算一次准确率
        print("accuracy:", sess.run(accuracy, feed_dict={x: iris_data_testStd, y_actual: iris_label_test}))