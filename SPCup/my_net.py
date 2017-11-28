import tensorflow as tf
def deepnn(x,phase):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1,28, 28, 1])
        tf.summary.image('input_image',x_image,max_outputs=10)

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    with tf.name_scope('pool'):
        h_pool2 = max_pool_3x3(h_conv2)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([14 * 14 * 64, 256])
        b_fc1 = bias_variable([256])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 14*14*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([256, 512])
        b_fc2 = bias_variable([512])
        h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
    with tf.name_scope('fc3'):
        W_fc3 = weight_variable([512, 10])
        b_fc3 = bias_variable([10])
        y_conv =tf.matmul(h_fc2 ,W_fc3) + b_fc3
    return y_conv

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
