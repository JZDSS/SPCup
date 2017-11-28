from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.losses as loss

from utils.input_pipeline import input_pipeline
from nets import twins

flags = tf.app.flags

flags.DEFINE_string('data_dir', '../patches', 'data direction')
flags.DEFINE_string('log_dir', './logs', 'log direction')
flags.DEFINE_string('ckpt_dir', './ckpt', 'check point direction')
flags.DEFINE_float('weight_decay', 0.0001, 'weight decay')
flags.DEFINE_float('momentum', 0.9, 'momentum')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('max_steps', 172000, 'max steps')
flags.DEFINE_integer('start_step', 1, 'start steps')
flags.DEFINE_string('model_name', 'model', '')
flags.DEFINE_string('gpu', '3', '')
flags.DEFINE_integer('blocks', 5, '')
flags.DEFINE_string('out_file', '', '')
flags.DEFINE_integer('patch_size', 64, '')
FLAGS = flags.FLAGS


def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    num_gpus = len(FLAGS.gpu.split(','))
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    if not tf.gfile.Exists(FLAGS.data_dir):
        raise RuntimeError('data direction is not exist!')

    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.ckpt_dir):
        tf.gfile.MakeDirs(FLAGS.ckpt_dir)
    train_example_batch, train_label_batch = input_pipeline(
        tf.train.match_filenames_once(os.path.join(FLAGS.data_dir, 'train', '*.tfrecords')), FLAGS.batch_size, FLAGS.patch_size)
    valid_example_batch, valid_label_batch = input_pipeline(
        tf.train.match_filenames_once(os.path.join(FLAGS.data_dir, 'valid', '*.tfrecords')), FLAGS.batch_size, FLAGS.patch_size)
    f = open(FLAGS.out_file, 'w')
    if not f:
        raise RuntimeError('OUTPUT FILE OPEN ERROR!!!!!!')
    with tf.name_scope('input'):
        x1 = tf.placeholder(tf.float32, [None, FLAGS.patch_size, FLAGS.patch_size, 3], 'x1')
        x2 = tf.placeholder(tf.float32, [None, FLAGS.patch_size, FLAGS.patch_size, 3], 'x2')
        tf.summary.image('show', x1, 1)
        tf.summary.image('show', x2, 1)

    with tf.name_scope('label'):
        y_1 = tf.placeholder(tf.int64, [None, 1], 'y1')
        y_2 = tf.placeholder(tf.int64, [None, 1], 'y2')
        y_ = tf.cast(tf.equal(y_1, y_2), tf.int64)

    is_training = tf.placeholder(tf.bool)

    y = twins.build_net(x1, x2, FLAGS.blocks, is_training)
    y = tf.nn.softmax(y)
    y_onehot = tf.reshape(tf.one_hot(y_, 2), [-1, 2])
    weights = tf.reshape(tf.cast(y_, tf.float32) * 10, [-1])
    with tf.name_scope('scores'):
        loss.mean_squared_error(y, y_onehot, weights)
        total_loss = tf.contrib.losses.get_total_loss(add_regularization_losses=True, name='total_loss')
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.reshape(tf.argmax(y, 1), [-1, 1]), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('train'):
        global_step = tf.Variable(FLAGS.start_step, name="global_step")
        # learning_rate = tf.train.piecewise_constant(global_step, [32000, 64000, 108000, ], [0.01, 0.001, 0.0001, 0.00001])
        learning_rate = tf.train.exponential_decay(0.01, global_step, 32000, 0.1)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.MomentumOptimizer(learning_rate, momentum=FLAGS.momentum).minimize(total_loss,
                                                                                                     global_step=global_step)
    tf.summary.scalar('lr', learning_rate)

    merged = tf.summary.merge_all()

    with tf.name_scope("saver"):
        saver = tf.train.Saver(name="saver")

    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
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
                x1s, y1s = get_batch(train_example_batch, train_label_batch)
                x2s, y2s = get_batch(train_example_batch, train_label_batch)
            else:
                x1s, y1s = get_batch(valid_example_batch, valid_label_batch)
                x2s, y2s = get_batch(valid_example_batch, valid_label_batch)
            return {x1: x1s, x2: x2s, y_1: y1s, y_2: y2s, is_training: on_training}

        for i in range(FLAGS.start_step, FLAGS.max_steps + 1):
            feed = feed_dict(True, True)
            sess.run(train_step, feed_dict=feed)
            if i % 1000 == 0 and i != 0:  # Record summaries and test-set accuracy
                loss0, acc0, summary = sess.run([total_loss, accuracy, merged], feed_dict=feed_dict(False, False))
                test_writer.add_summary(summary, i)
                loss1, acc1, summary = sess.run([total_loss, accuracy, merged], feed_dict=feed_dict(True, False))
                train_writer.add_summary(summary, i)
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file=f)
                print('step %d: train_acc=%f, train_loss=%f; test_acc=%f, test_loss=%f' % (i, acc1, loss1, acc0, loss0),
                      file=f)
                saver.save(sess, os.path.join(FLAGS.ckpt_dir, FLAGS.model_name))
                f.flush()

        coord.request_stop()
        coord.join(threads)

    train_writer.close()
    test_writer.close()
    f.close()


if __name__ == '__main__':
    tf.app.run()

