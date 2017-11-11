import tensorflow as tf
import tensorflow.contrib.losses as loss
import numpy as np
import os

import resnet as res


def read_from_tfrecord(tfrecord_file_queue):
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'patch_raw': tf.FixedLenFeature([], tf.string)
        }, name='features')
    image = tf.decode_raw(tfrecord_features['patch_raw'], tf.uint8)
    ground_truth = tf.decode_raw(tfrecord_features['label'], tf.int32)

    image = tf.cast(tf.reshape(image, [64, 64, 3]), tf.float32)
    image = tf.image.per_image_standardization(image)
    ground_truth = tf.reshape(ground_truth, [1])
    return image, ground_truth

def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    example, label = read_from_tfrecord(filename_queue)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch


flags = tf.app.flags

flags.DEFINE_string('data_dir', '../patches', 'data direction')
flags.DEFINE_string('log_dir', './logs', 'log direction')
flags.DEFINE_string('ckpt_dir', './ckpt', 'check point direction')
flags.DEFINE_float('weight_decay', 0.0001, 'weight decay')
flags.DEFINE_float('momentum', 0.9, 'momentum')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.app.flags.DEFINE_integer('max_steps', 64000, 'max steps')
tf.app.flags.DEFINE_integer('start_step', 1, 'start steps')
flags.DEFINE_string('model_name', 'model', '')

FLAGS = flags.FLAGS


def main(_):

    if not tf.gfile.Exists(FLAGS.data_dir):
        raise RuntimeError('data direction is not exist!')

    # if tf.gfile.Exists(FLAGS.log_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.log_dir)
    # tf.gfile.MakeDirs(FLAGS.log_dir)

    train_example_batch, train_label_batch = input_pipeline([FLAGS.data_dir + '/spc_train.tfrecords'], FLAGS.batch_size)
    valid_example_batch, valid_label_batch = input_pipeline([FLAGS.data_dir + '/spc_valid.tfrecords'], FLAGS.batch_size)

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 64, 64, 3], 'x')
        tf.summary.image('show', x, 1)

    with tf.name_scope('label'):
        y_ = tf.placeholder(tf.int64, [None, 1], 'y')
    is_training = tf.placeholder(tf.bool)
    with tf.variable_scope('net'):
        y = res.build_net(x, 3, is_training)

    with tf.name_scope('scores'):
        loss.sparse_softmax_cross_entropy(y, y_, scope='cross_entropy')
        total_loss = tf.contrib.losses.get_total_loss(add_regularization_losses=True, name='total_loss')
        tf.summary.scalar('loss', total_loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.reshape(tf.argmax(y, 1), [-1, 1]), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('train'):
        global_step = tf.Variable(FLAGS.start_step, name="global_step")
        learning_rate = tf.train.piecewise_constant(global_step, [32000, 48000], [0.1, 0.01, 0.001])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.MomentumOptimizer(learning_rate, momentum=FLAGS.momentum).minimize(
                total_loss, global_step=global_step)
    tf.summary.scalar('lr', learning_rate)

    merged = tf.summary.merge_all()

    with tf.name_scope("saver"):
        saver = tf.train.Saver(name="saver")

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        if tf.gfile.Exists(os.path.join(FLAGS.ckpt_dir, 'checkpoint')):
            saver.restore(sess, os.path.join(FLAGS.ckpt_dir, FLAGS.model_name))
        else:
            sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)
        train_writer.flush()
        test_writer.flush()

        def feed_dict(train, on_training):
            def get_batch(data, labels):
                d, l = sess.run([data, labels])
                d = d.astype(np.float32)
                l = l.astype(np.int64)
                return d, l

            if train:
                xs, ys = get_batch(train_example_batch, train_label_batch)
            else:
                xs, ys = get_batch(valid_example_batch, valid_label_batch)
            return {x: xs, y_: ys, is_training: on_training}

        for i in range(FLAGS.start_step, FLAGS.max_steps + 1):
            sess.run(train_step, feed_dict=feed_dict(True, True))
            if i % 100 == 0 and i != 0:  # Record summaries and test-set accuracy
                acc0, summary = sess.run([accuracy, merged], feed_dict=feed_dict(False, False))
                test_writer.add_summary(summary, i)
                acc1, summary = sess.run([accuracy, merged], feed_dict=feed_dict(True, False))
                train_writer.add_summary(summary, i)
                print('step %d: train_acc=%f; test_acc=%f' % (i, acc1, acc0))
                saver.save(sess, os.path.join(FLAGS.ckpt_dir, FLAGS.model_name))

        coord.request_stop()
        coord.join(threads)

    train_writer.close()
    test_writer.close()


if __name__ == '__main__':
    tf.app.run()

