# 神经网络——手写数字识别
# 使用这一行来下载代码
# one_hot就是数据的标签形式，例如5就是5,[0,0,0,0,0,1,0,0,0,0,0]
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 下载并加载数据
mnist = input_data.read_data_sets("./mnist_data/", one_hot=True)

print(mnist.train)

# 数据与标签的占位
# 输入值，输入层的784个神经元，可以批量输入，None表示输入样本的数量暂不确定
x = tf.placeholder(tf.float32, shape=[None, 784])
# 输出值，输出层的10个神经元，可以批量输入，使用None表示输入样本的数量暂不确定
# 这里输入的是真实值。后面和预测值比较生成损失函数
y_actual = tf.placeholder(tf.float32, shape=[None, 10])

# 初始化权重和偏置，这里是需要更新的
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# tf.matmul(x,W)+b 这是矩阵乘法，就是神经网络的前向传播
# softmax回归，得到预测概率，这是预测值
y_predict = tf.nn.softmax(tf.matmul(x, W) + b)

# 预测值和真实值，通过交叉熵函数，得到损失函数
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_predict, labels=tf.argmax(y_actual, 1))

# 梯度下降法使得残差最小
# 0.01是学习率，使用梯度下降算法，更新W和b，最小化损失函数
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 测试阶段，测试准确度计算，1也是表示1这个维度，就是矩阵横轴这个维度
# [[0~9],-->一个样本
# [0~9],]
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))  # 多个批次的准确度均值

# 初始化值的变量还没有真正运行，只是搭建了一个计算图，使用这个方法就是运行了
# tf.zeros变量初始化为0，先运行成功
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # 训练，迭代1000次
    for i in range(1000):
        # 训练集批量喂入
        batch_xs, batch_ys = mnist.train.next_batch(100)  # 按批次训练，每批100行数据
        # 执行梯度下降算法，每执行一次，参数更新一次
        sess.run(train_step, feed_dict={x: batch_xs, y_actual: batch_ys})  # 执行训练
        if (i % 100 == 0):  # 每训练100次，测试一次
            # 打印准确率：如果喂入的是训练集，计算出的就是训练集的准确率
            print("accuracy:", sess.run(accuracy, feed_dict={x: mnist.train.images, y_actual: mnist.train.labels}))
            # 打印准确率：如果喂入的是测试集，计算出的就是测试集的准确率
            print("accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_actual: mnist.test.labels}))

# 分类问题，
# 单分类，
# 结果是"是否"，0～1， 真实值0或1，预测值0～1，
# 神经网络，只有一个输出值，激活函数就用sigmoid

# 多分类
# 真实值[0,0,1]  预测值[0.1,0.1,0.8]
# softmax函数

# 【项目四】A 90分以上
# iris，其他数据集
# 1.数据集要分成训练集、测试集
# 2.打印出测试集的准确率
# 3。每行代码自己写解释，不允许抄袭
