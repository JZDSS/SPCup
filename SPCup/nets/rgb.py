from nets import resnet
import tensorflow as tf
import tensorflow.contrib.layers as layers
import math

def build_net(x, n, is_training, FLAGS):
    R, G, B = tf.split(x, 3, 3)
    with tf.variable_scope('R'):
        R = resnet.build_net(R, n, is_training, FLAGS)
    with tf.variable_scope('G'):
        G = resnet.build_net(G, n, is_training, FLAGS)
    with tf.variable_scope('B'):
        B = resnet.build_net(B, n, is_training, FLAGS)
    h = tf.concat([R, G, B], axis=1)
    h = layers.fully_connected(h, num_outputs=FLAGS.num_classes, activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(
                stddev=math.sqrt(2.0 / 30 / FLAGS.num_classes)),
                weights_regularizer=layers.l2_regularizer(FLAGS.weight_decay),
                normalizer_fn=layers.batch_norm,
                normalizer_params={'is_training': is_training}, scope='fc')
    return h

