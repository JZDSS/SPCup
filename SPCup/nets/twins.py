import tensorflow as tf
import tensorflow.contrib.layers as layers
import math


def block(inputs, num_outputs, weight_decay, scope, is_training, down_sample = False):
    with tf.variable_scope(scope):
        num_inputs = inputs.get_shape().as_list()[3]

        res = layers.conv2d(inputs, num_outputs = num_outputs, kernel_size=[3, 3], stride=2 if down_sample else 1,
                            weights_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/9.0/num_inputs)),
                            weights_regularizer=layers.l2_regularizer(weight_decay),
                            scope='conv1', normalizer_fn=layers.batch_norm,
                            normalizer_params={'is_training': is_training})

        res = layers.conv2d(res, num_outputs=num_outputs, kernel_size=[3, 3], activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/9.0/num_outputs)),
                            weights_regularizer=layers.l2_regularizer(weight_decay),
                            scope='conv2', normalizer_fn=layers.batch_norm,
                            normalizer_params={'is_training': is_training})
        if num_inputs != num_outputs:
            inputs = layers.conv2d(inputs, num_outputs=num_outputs, kernel_size=[1, 1], activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(
                                    stddev=math.sqrt(2.0 /  num_outputs)),
                                weights_regularizer=layers.l2_regularizer(weight_decay),
                                scope='short_cut', stride=[2, 2], normalizer_fn=layers.batch_norm,
                                normalizer_params={'is_training': is_training})
        res = tf.nn.relu(res + inputs)

    return res

def build_net(x1, x2, n, is_training):
    shape = x1.get_shape().as_list()
    with tf.variable_scope('left'):
        with tf.variable_scope('pre'):
            pre = layers.conv2d(inputs=x1, num_outputs=16, kernel_size=[3, 3], scope='conv',
                                weights_initializer=tf.truncated_normal_initializer(
                                    stddev=math.sqrt(2.0 / 9.0 / shape[3])),
                                weights_regularizer=layers.l2_regularizer(0.0001),
                                normalizer_fn=layers.batch_norm, normalizer_params={'is_training': is_training})
        h = pre
        for i in range(1, n + 1):
            h = block(h, 16, 0.0001, '16_block{}'.format(i), is_training)

        h = block(h, 32, 0.0001, '32_block1', is_training, True)
        for i in range(2, n + 1):
            h = block(h, 32, 0.0001, '32_block{}'.format(i), is_training)
    h1 = h

    with tf.variable_scope('right'):
        with tf.variable_scope('pre'):
            pre = layers.conv2d(inputs=x2, num_outputs=16, kernel_size=[3, 3], scope='conv',
                                weights_initializer=tf.truncated_normal_initializer(
                                    stddev=math.sqrt(2.0 / 9.0 / shape[3])),
                                weights_regularizer=layers.l2_regularizer(0.0001),
                                normalizer_fn=layers.batch_norm, normalizer_params={'is_training': is_training})
        h = pre
        for i in range(1, n + 1):
            h = block(h, 16, 0.0001, '16_block{}'.format(i), is_training)

        h = block(h, 32, 0.0001, '32_block1', is_training, True)
        for i in range(2, n + 1):
            h = block(h, 32, 0.0001, '32_block{}'.format(i), is_training)
    h2 = h

    h = tf.concat([h1, h2], axis=3)

    # h = block(h, 128, 0.0001, '128_block1', is_training, True)
    for i in range(1, n + 1):
        h = block(h, 64, 0.0001, '128_block{}'.format(i), is_training)

    shape = h.get_shape().as_list()

    h = layers.avg_pool2d(h, [shape[1], shape[2]], scope='global_pool')
    h = layers.conv2d(inputs=h, num_outputs=2, kernel_size=[1, 1], scope='fc1', padding='VALID',
                      weights_initializer=tf.truncated_normal_initializer(
                          stddev=math.sqrt(2.0 / 64 / 10)),
                      weights_regularizer=layers.l2_regularizer(0.0001),
                      normalizer_fn=layers.batch_norm,
                      normalizer_params={'is_training': is_training})

    return tf.reshape(h, [-1, 2])


