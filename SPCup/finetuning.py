import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('logdir', './logs', 'Log direction')
flags.DEFINE_string('ckpt', './ckpt', 'Check point direction')
flags.DEFINE_float('lr', 0.001, 'Learning rate')
flags.DEFINE_integer('batch', 100, 'Batch size')
flags.DEFINE_integer('epoch', 100, 'Number of epochs to train')

FLAGS = flags.FLAGS


def main(_):
    pass


if __name__ == "__main__":
    tf.app.run()