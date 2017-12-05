import tensorflow as tf


def net(x, FLAGS, is_training, num_classes):
    type = FLAGS.type
    if type == 'resnet':
        from nets import resnet as my_net
    elif type == 'slim':
        from nets import slim as my_net
    elif type == 'rgb':
        from nets import rgb as my_net
    elif type == 'twins':
        from nets import twins as my_net
    elif type == 'resnet5':
        from nets import resnet5 as my_net
    elif type == 'resnet7':
        from nets import resnet7 as my_net
    elif type == 'split':
        from nets import split as my_net
    else:
        raise RuntimeError('Type error!!')

    with tf.variable_scope('net'):
        y = my_net.build_net(x, FLAGS.blocks, is_training, num_classes)
    return y