import tensorflow as tf
import tensorflow.contrib.losses as loss
import tensorflow.contrib.layers as layers
import numpy as np
import os
import pickle
import time

import resnet as res


def read_from_tfrecord(tfrecord_file_queue):
    # tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
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
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch



flags = tf.app.flags

flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
flags.DEFINE_string('data_dir', '../patches', 'data direction')
flags.DEFINE_string('log_dir', './logs', 'log direction')
flags.DEFINE_string('ckpt_dir', './ckpt', 'check point direction')
flags.DEFINE_float('weight_decay', 0.0001, 'weight decay')
flags.DEFINE_integer('decay_steps', 100, 'decay steps')
flags.DEFINE_float('decay_rate', 0.95, 'decay rate')
flags.DEFINE_float('momentum', 0.9, 'momentum')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.app.flags.DEFINE_float('dropout', 0.5, 'keep probability')
tf.app.flags.DEFINE_integer('max_steps', 64000, 'max steps')
tf.app.flags.DEFINE_integer('start_step', 1, 'start steps')

FLAGS = flags.FLAGS


def main(_):

    # if not tf.gfile.Exists(FLAGS.data_dir):
    #     print('data direction is not exist!')
    #     return -1

    # if tf.gfile.Exists(FLAGS.log_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.log_dir)
    # tf.gfile.MakeDirs(FLAGS.log_dir)

    # train_data, train_labels = load_train_data()
    # # name = 'cifar10_train'
    #
    # valid_data, valid_labels = load_valid_data()
    # # name = 'cifar10_valid'
    # train_data = (train_data - 128) / 128.0
    # valid_data = (valid_data - 128) / 128.0

    train_example_batch, train_label_batch = input_pipeline(['../patches/spc_train.tfrecords'], FLAGS.batch_size)
    valid_example_batch, valid_label_batch = input_pipeline(['../patches/spc_valid.tfrecords'], FLAGS.batch_size)

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 64, 64, 3], 'x')
        tf.summary.image('show', x, 1)

    with tf.name_scope('label'):
        y_ = tf.placeholder(tf.int64, [None, 1], 'y')

    with tf.variable_scope('net'):
        # y, keep_prob = build_net(x)
        y, keep_prob = res.build_net(x, 3)

    with tf.name_scope('scores'):
        loss.sparse_softmax_cross_entropy(y, y_, scope='cross_entropy')
        total_loss = tf.contrib.losses.get_total_loss(add_regularization_losses=True, name='total_loss')

        # expp = tf.exp(y)
        #
        # correct = tf.reduce_sum(tf.multiply(tf.one_hot(y_, 10), y), 1)
        #
        # total_loss = total_loss + tf.reduce_mean(tf.log(tf.reduce_sum(expp, 1)), 0) - tf.reduce_mean(correct, 0)

        tf.summary.scalar('loss', total_loss)
        # with tf.name_scope('accuracy'):
        # with tf.name_scope('correct_prediction'):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.reshape(tf.argmax(y, 1), [-1, 1]), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        # accuracy = tf.metrics.accuracy(labels=y_, predictions=tf.argmax(y, 1), name='accuracy')
        # tf.summary.scalar('accuracy', accuracy)

    # loss.mean_squared_error(predictions, labels, scope='l2_1')
    # loss.mean_squared_error(predictions, labels, scope='l2_2')

    # loss_collect = tf.get_collection(tf.GraphKeys.LOSSES)
    # print((tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
    with tf.name_scope('train'):
        global_step = tf.Variable(FLAGS.start_step, name="global_step")
        # learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
        #     global_step, FLAGS.decay_steps, FLAGS.decay_rate, True, "learning_rate")
        learning_rate = tf.train.piecewise_constant(global_step, [32000, 48000], [0.1, 0.01, 0.001])
        train_step = tf.train.MomentumOptimizer(learning_rate, momentum=FLAGS.momentum).minimize(
            total_loss, global_step=global_step)
    tf.summary.scalar('lr', learning_rate)

    merged = tf.summary.merge_all()

    with tf.name_scope("saver"):
        saver = tf.train.Saver(name="saver")

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver = tf.train.Saver(name="saver")

        if tf.gfile.Exists(os.path.join(FLAGS.ckpt_dir, 'checkpoint')):
            saver.restore(sess, os.path.join(FLAGS.ckpt_dir, 'model.ckpt'))
        else:
            sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test', sess.graph)
        train_writer.flush()
        test_writer.flush()

        def feed_dict(train, kk=FLAGS.dropout):
            """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""

            def get_batch(data, labels):
                # id = np.random.randint(low=0, high=labels.shape[0], size=FLAGS.batch_size, dtype=np.int32)
                # return data[id, ...], labels[id]
                d, l = sess.run([data, labels])
                d = d.astype(np.float32)
                l = l.astype(np.int64)
                return d, l


            if train:
                xs, ys = get_batch(train_example_batch, train_label_batch)
                # xs = tmp
                # tmp = np.pad(tmp, 4, 'constant')
                # for ii in range(FLAGS.batch_size):
                #     xx = np.random.randint(0, 9)
                #     yy = np.random.randint(0, 9)
                #     xs[ii,:] = np.fliplr(tmp[ii + 4,xx:xx + 32, yy:yy + 32,4:7]) if np.random.randint(0, 2) == 1 \
                #         else tmp[ii + 4,xx:xx + 32, yy:yy + 32,4:7]
                k = kk
            else:
                xs, ys = get_batch(valid_example_batch, valid_label_batch)
                # xs = valid_data
                # ys = valid_labels
                k = 1.0
            return {x: xs, y_: ys, keep_prob: k}

        for i in range(FLAGS.start_step, FLAGS.max_steps + 1):
            sess.run(train_step, feed_dict=feed_dict(True))
            if i % 100 == 0 and i != 0:  # Record summaries and test-set accuracy
                acc, summary = sess.run([accuracy, merged], feed_dict=feed_dict(False))
                # test_writer.add_summary(summary, i)
                print(i)
                print(acc)
                acc, summary = sess.run([accuracy, merged], feed_dict=feed_dict(True))
                # train_writer.add_summary(summary, i)
                print(acc)
                saver.save(sess, os.path.join(FLAGS.ckpt_dir, 'model.ckpt'))

        coord.request_stop()
        coord.join(threads)


    train_writer.close()
    test_writer.close()


if __name__ == '__main__':
    tf.app.run()

