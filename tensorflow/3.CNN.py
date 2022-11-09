import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/",one_hot=True)

x = tf.placeholder(tf.float32, [None, 784], name='x-input')
y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

# 初始化可更新的参数（权重W），使用截断的正态分布生成随机数
# 输入的是Shape，返回值是tf.variable,tf.truncated_normal这是初始值，之后要更新的
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01,seed=1234)
    return tf.Variable(initial)

# 初始化可更新的偏置值（偏置值b），赋值为0.1
# 输入的是Shape，返回值是tf.variable,tf.constant这是初始值，之后要更新的
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# SAME表示输入和输出是同一尺寸
# 对于图片，因为只有两维，通常strides取[1，stride，stride，1]
# x是输入图片，有可能是3维的RGB原始图片，也可以是n维特征图
# w是卷积核，卷积核的维度必须与x一致，一个卷积核产生一张特征图，n个卷积核产生n张特征图
# 输出的时候这n张特征图作为一张新的特征图来处理
def conv2d(x, w):
    # padding=SAME全0填充后，输出的图像长宽不变
    # [1固定，图片垂直方向步长，图片水平方向步长，1固定]
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# ksize是池化窗口的大小，strides是池化过程的步长
def max_pool_2x2(x):
    # [1固定值, 2窗口长, 2窗口宽, 1固定值]
    # [1固定值, 2垂直方向步长, 2水平方向步长, 1固定值]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 第一层卷积 前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目
w_conv1 = weight_variable([5, 5, 1, 32]) # [卷积核的长，卷积核的宽，卷积核的维度，卷积核的数量]

# 第一层卷积核的偏置项
b_conv1 = bias_variable([32])

# reshape中的-1表示-1代表的含义是不用我们自己指定这一维的大小，
# 函数会自动计算，但列表中只能存在一个-1,
# 其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积 前两个维度是patch的大小，接着是输入的通道数目，最后输出的通道数目
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积层 因为第一层输出的通道数目是32 故第二层输入的通道数目也应该为32
# [卷积核的长，卷积核的宽，卷积核的维度，卷积核的数量] 卷积核的维度和输入图像的维度保持一致
w_conv2 = weight_variable([5, 5, 32, 64])  # 这里的64是自己给定的,可以为其他的值
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层,假设全连接层的个数为1024，当然也可以设置成其他值
# 两次池化后尺寸由28*28 -> 14*14 ->7*7
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# 扁平化
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
# drop_out
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# 训练和评估模型  !!!注意cross_entropy应该加负号！！
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
# train_step=tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#x - input
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(2000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0]
                , y_: batch[1], keep_prob: 0.5})
            print("after %d steps the trainnig accuracy is %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # http://blog.csdn.net/zsg2063/article/details/74332487是可能遇到的问题的答案 全部的testdata不能通过一次计算算出accuracy
    # print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))

    for i in range(10):
        testSet = mnist.test.next_batch(1000)
        print("test accuracy %g" % accuracy.eval(feed_dict={x: testSet[0], y_: testSet[1], keep_prob: 0.5}))
