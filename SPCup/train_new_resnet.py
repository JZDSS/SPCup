from __future__ import print_function
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.losses as loss
from nets import build

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
flags.DEFINE_string('type', '', '')
FLAGS = flags.FLAGS



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

    image = tf.cast(tf.reshape(image, [FLAGS.patch_size, FLAGS.patch_size, 3]), tf.float32)
    image = tf.image.per_image_standardization(image)
    ground_truth = tf.reshape(ground_truth, [1])
    return image, ground_truth

def input_pipeline(filenames, batch_size, read_threads=2, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
    example_list = [read_from_tfrecord(filename_queue)
                  for _ in range(read_threads)]
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch_join(
      example_list, batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch


def main(_):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
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
        tf.train.match_filenames_once(os.path.join(FLAGS.data_dir, 'train', '*.tfrecords')), FLAGS.batch_size)
    valid_example_batch, valid_label_batch = input_pipeline(
        tf.train.match_filenames_once(os.path.join(FLAGS.data_dir, 'valid', '*.tfrecords')), FLAGS.batch_size)
    f = open(FLAGS.out_file, 'w')
    if not f:
        raise RuntimeError('OUTPUT FILE OPEN ERROR!!!!!!')
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, FLAGS.patch_size, FLAGS.patch_size, 3], 'x')
        tf.summary.image('show', x, 1)

    with tf.name_scope('label'):
        y_ = tf.placeholder(tf.int64, [None, 1], 'y')
    
    is_training = tf.placeholder(tf.bool)

    y = build.net(x, FLAGS, is_training)

    with tf.name_scope('scores'):
        loss.sparse_softmax_cross_entropy(y, y_, scope='cross_entropy')
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
            train_step = tf.train.MomentumOptimizer(learning_rate, momentum=FLAGS.momentum).minimize(total_loss, global_step=global_step)
    tf.summary.scalar('lr', learning_rate)

    merged = tf.summary.merge_all()

    with tf.name_scope("saver"):
        saver = tf.train.Saver(name="saver")

    with tf.Session(config = config) as sess:
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
                xs, ys = get_batch(train_example_batch, train_label_batch)
            else:
                xs, ys = get_batch(valid_example_batch, valid_label_batch)
            return {x: xs, y_: ys, is_training: on_training}

        for i in range(FLAGS.start_step, FLAGS.max_steps + 1):
            feed = feed_dict(True, True)
            sess.run(train_step, feed_dict=feed)
            if i % 1000 == 0 and i != 0:  # Record summaries and test-set accuracy
                loss0, acc0, summary = sess.run([total_loss, accuracy, merged], feed_dict=feed_dict(False, False))
                test_writer.add_summary(summary, i)
                # feed[is_training] = FLAGS
                loss1, acc1, summary = sess.run([total_loss, accuracy, merged], feed_dict=feed_dict(True, False))
                train_writer.add_summary(summary, i)
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file=f)
                print('step %d: train_acc=%f, train_loss=%f; test_acc=%f, test_loss=%f' % (i, acc1, loss1, acc0, loss0), file=f)
                saver.save(sess, os.path.join(FLAGS.ckpt_dir, FLAGS.model_name))
                f.flush()

        coord.request_stop()
        coord.join(threads)

    train_writer.close()
    test_writer.close()
    f.close()


if __name__ == '__main__':
    tf.app.run()

