import tensorflow.contrib.layers as layers
import tensorflow as tf
import math


def block(inputs, num_outputs, weight_decay, scope, is_training, down_sample = False):
    with tf.variable_scope(scope):

        num_inputs = inputs.get_shape().as_list()[3]

        res = layers.conv2d(inputs, num_outputs = num_outputs, kernel_size=[1, 1], stride=2 if down_sample else 1,
                            weights_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/num_inputs)),
                            weights_regularizer=layers.l2_regularizer(weight_decay),
                            scope='conv1', normalizer_fn=layers.batch_norm,
                            normalizer_params={'is_training': is_training})

        res = layers.conv2d(res, num_outputs=num_outputs, kernel_size=[1, 1], activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/num_outputs)),
                            weights_regularizer=layers.l2_regularizer(weight_decay),
                            scope='conv2', normalizer_fn=layers.batch_norm,
                            normalizer_params={'is_training': is_training})
        if  num_inputs != num_outputs:
            # w = tf.Variable(tf.truncated_normal([1, 1, num_inputs, num_outputs], stddev=math.sqrt(2.0/num_inputs)))
            # inputs = tf.nn.conv2d(inputs, w, [1, 2, 2, 1], 'SAME')
            inputs = layers.conv2d(inputs, num_outputs=num_outputs, kernel_size=[1, 1], activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(
                                    stddev=math.sqrt(2.0 /  num_outputs)),
                                weights_regularizer=layers.l2_regularizer(weight_decay),
                                # biases_regularizer=layers.l2_regularizer(weight_decay),
                                scope='short_cut', stride=[2, 2], normalizer_fn=layers.batch_norm, 
                                normalizer_params={'is_training': is_training})
        res = tf.nn.relu(res + inputs)

    return res


def build_net(x, n, is_training, FLAGS):
    with tf.variable_scope('pre'):
        pre = layers.conv2d(inputs=x, num_outputs=16,  kernel_size = [3, 3], scope='conv1',
                            weights_initializer=tf.truncated_normal_initializer(
                            stddev=math.sqrt(2.0 / 9.0 / 3)),
                            weights_regularizer=layers.l2_regularizer(FLAGS.weight_decay),
                            normalizer_fn=layers.batch_norm, normalizer_params={'is_training': is_training})
        # pre = layers.max_pool2d(pre, [2, 2], padding='SAME', scope='pool')
    h = pre
    for i in range(1, n + 1):
        h = block(h, 16, FLAGS.weight_decay, '16_block{}'.format(i), is_training)
	
    h = layers.conv2d(inputs=h, num_outputs=32,  kernel_size = [3, 3], scope='conv2',
                            weights_initializer=tf.truncated_normal_initializer(
                            stddev=math.sqrt(2.0 / 9.0 / 32)), stride=[2, 2],
                            weights_regularizer=layers.l2_regularizer(FLAGS.weight_decay),
                            normalizer_fn=layers.batch_norm, normalizer_params={'is_training': is_training})
    # h = block(h, 32, 0.0001, '32_block1', is_training, True)
    for i in range(2, n + 1):
        h = block(h, 32, FLAGS.weight_decay, '32_block{}'.format(i), is_training)
	
    h = layers.conv2d(inputs=h, num_outputs=64,  kernel_size = [3, 3], scope='conv3',
                            weights_initializer=tf.truncated_normal_initializer(
                            stddev=math.sqrt(2.0 / 9.0 / 64)), stride=[2, 2],
                            weights_regularizer=layers.l2_regularizer(FLAGS.weight_decay),
                            normalizer_fn=layers.batch_norm, normalizer_params={'is_training': is_training})
    # h = block(h, 64, 0.0001, '64_block1', is_training, True)
    for i in range(2, n + 1):
        h = block(h, 64, FLAGS.weight_decay, '64_block{}'.format(i), is_training)

    shape = h.get_shape().as_list()

    h = layers.avg_pool2d(h, [shape[1], shape[2]], scope='global_pool')
    h = layers.conv2d(inputs=h, num_outputs=FLAGS.num_classes, kernel_size=[1, 1], scope='fc1', padding='VALID',
                  weights_initializer=tf.truncated_normal_initializer(
                      stddev=math.sqrt(2.0 / 64 / FLAGS.num_classes)),
                  weights_regularizer=layers.l2_regularizer(FLAGS.weight_decay),
                  normalizer_fn=layers.batch_norm, activation_fn=None,
                  normalizer_params={'is_training': is_training})

    return tf.reshape(h, [-1, FLAGS.num_classes])
