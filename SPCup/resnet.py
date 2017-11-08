import tensorflow.contrib.layers as layers
import tensorflow as tf
import math
def block(inputs, num_outputs, weight_decay, scope, down_sample = False):
    with tf.variable_scope(scope):

        num_inputs = inputs.get_shape().as_list()[3]

        res = layers.conv2d(inputs, num_outputs = num_outputs, kernel_size=[3, 3], stride=2 if down_sample else 1,
                            weights_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/9.0/num_inputs)),
                            weights_regularizer=layers.l2_regularizer(weight_decay),
                            biases_regularizer=layers.l2_regularizer(weight_decay),
                            scope='conv1', normalizer_fn=layers.batch_norm)

        res = layers.conv2d(res, num_outputs=num_outputs, kernel_size=[3, 3], activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/9.0/num_outputs)),
                            weights_regularizer=layers.l2_regularizer(weight_decay),
                            biases_regularizer=layers.l2_regularizer(weight_decay),
                            scope='conv2', normalizer_fn=layers.batch_norm)
        if  num_inputs != num_outputs:
            w = tf.Variable(tf.truncated_normal([1, 1, num_inputs, num_outputs], stddev=math.sqrt(2.0/num_inputs)))
            inputs = tf.nn.conv2d(inputs, w, [1, 2, 2, 1], 'SAME')
        res = tf.nn.relu(res + inputs)

    return res


def build_net(x, n):
    with tf.variable_scope('pre'):
        pre = layers.conv2d(inputs=x, num_outputs=16,  kernel_size = [3, 3], scope='conv',
                            weights_initializer=tf.truncated_normal_initializer(
                            stddev=math.sqrt(2.0 / 9.0 / 3)),
                            weights_regularizer=layers.l2_regularizer(0.0001),
                            biases_regularizer=layers.l2_regularizer(0.0001),
                            normalizer_fn=layers.batch_norm)
        # pre = layers.max_pool2d(pre, [2, 2], padding='SAME', scope='pool')
    h = pre
    for i in range(1, n + 1):
        h = block(h, 16, 0.0001, '16_block{}'.format(i))

    h = block(h, 32, 0.0001, '32_block1', True)
    for i in range(2, n + 1):
        h = block(h, 32, 0.0001, '32_block{}'.format(i))

    h = block(h, 64, 0.0001, '64_block1', True)
    for i in range(2, n + 1):
        h = block(h, 64, 0.0001, '64_block{}'.format(i))

    # h = layers.avg_pool2d(h, [8, 8])

    # h = layers.fully_connected(inputs=h, num_outputs=10, activation_fn=None,
    #                            weights_initializer=tf.truncated_normal_initializer(stddev=0.05),
    #                            weights_regularizer=layers.l2_regularizer(0.0001),
    #                            biases_regularizer=layers.l2_regularizer(0.0001),
    #                            scope='fc1')
    h = layers.avg_pool2d(h, [16, 16], scope='global_pool')
    h = layers.conv2d(inputs=h, num_outputs=10, kernel_size=[1, 1], scope='fc1', padding='VALID',
                  weights_initializer=tf.truncated_normal_initializer(
                      stddev=math.sqrt(2.0 / 64 / 10)),
                  weights_regularizer=layers.l2_regularizer(0.0001),
                  biases_regularizer=layers.l2_regularizer(0.0001),
                  normalizer_fn=layers.batch_norm)
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    return tf.reshape(h, [-1, 10]), keep_prob
