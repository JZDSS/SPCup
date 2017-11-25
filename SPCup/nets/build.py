import tensorflow as tf

def net(x, FLAGS, is_training):
    type = FLAGS.type
    if type == 'resnet':
        from nets import resnet as my_net
    elif type == 'slim':
        from nets import slim as my_net
    elif type == 'rgb':
        from nets import rgb as my_net
    else:
        raise RuntimeError('Type error!!')

    with tf.variable_scope('net'):
        y = my_net.build_net(x, FLAGS.blocks, is_training)
    return y