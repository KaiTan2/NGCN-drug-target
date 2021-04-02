import tensorflow as tf
from tflearn.activations import relu

def weight_variable(shape): #根据输入的大小来生成权重矩阵
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32) #从截断的正态分布中输出随机值
    return tf.Variable(initial, dtype=tf.float32)

def bias_variable(shape): #同理定义初始值为0.1，大小为shape
    initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)

def a_layer(x, units):#初始化权重矩阵，根据输入的x值进行计算，返回激活函数后的结果
    W = weight_variable([x.get_shape().as_list()[1], units]) #首先返回矩阵大小，然后生成一个行列数的列表，返回列数
    b = bias_variable([units])
    tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W)) #规则化过程，防止过度拟合，把值加入到图中
    return relu(tf.matmul(x, W) + b) #矩阵乘法


def bi_layer(x0,x1,sym,dim_pred):#感觉这个是下一层？不，重构网络
    if sym == False: #判断条件还不知道
        W0p = weight_variable([x0.get_shape().as_list()[1],dim_pred])#生成（x0特征数，dim_pred）大小的权值矩阵
        W1p = weight_variable([x1.get_shape().as_list()[1],dim_pred])#生成（x1特征数，dim_pred）大小的权值矩阵
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0p)) #规则化后加入到图中
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W1p)) #规则化后加入到图中
        return tf.matmul(tf.matmul(x0, W0p),
                            tf.matmul(x1, W1p), transpose_b=True) #返回矩阵相乘的结果
    else:
        W0p = weight_variable([x0.get_shape().as_list()[1],dim_pred])
        tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(W0p))
        return tf.matmul(tf.matmul(x0, W0p),
                            tf.matmul(x1, W0p),transpose_b=True) #判断条件的区别，是否使用相同的权重矩阵
