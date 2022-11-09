import tensorflow as tf

# 张量：多维数组
# 计算图：所有的数学公式都是计算图，搭建神经网络
# 会话：计算图只有放入会话中才能算出实际的数据

# 定义张量
"""
1. tf.constant定义常数，无法被修改，不参与优化。
2. tf.Variable定义变量，神经网络优化中参与优化，每次迭代都要被更新
3. tf.placeholder用来占位，暂时没有数据，等网络搭建好之后，把数据集喂入
"""
a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([2.0, 3.0], name='b')

# 所有的计算都是在搭建计算图
result = a + b
print(result)

# 这个方法不常用，因为经常忘记close，会影响计算资源
#创建会话
sess = tf.Session()
print(sess.run(a)) # 计算张量
#关闭会话
sess.close()

# 使用这种方法创建会话，这个会自动close
with tf.Session() as sess:
    print(sess.run(a))  # 计算张量
