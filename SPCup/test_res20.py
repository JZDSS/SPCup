import tensorflow as tf
import tensorflow.contrib.losses as loss
import tensorflow.contrib.layers as layers
import numpy as np
import os
import pickle
import time
import matplotlib.pyplot as plt

import resnet as res

flags = tf.app.flags

flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
flags.DEFINE_string('data_dir', '../data', 'data direction')
flags.DEFINE_string('log_dir', './logs', 'log direction')
flags.DEFINE_string('ckpt_dir', '.res20/ckpt', 'check point direction')
flags.DEFINE_float('weight_decay', 0.0001, 'weight decay')
flags.DEFINE_integer('decay_steps', 100, 'decay steps')
flags.DEFINE_float('decay_rate', 0.95, 'decay rate')
flags.DEFINE_float('momentum', 0.9, 'momentum')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_float('dropout', 0.5, 'keep probability')
flags.DEFINE_integer('max_steps', 64000, 'max steps')
flags.DEFINE_integer('start_step', 1, 'start steps')
flags.DEFINE_bool('verbose', False, '')
flags.DEFINE_integer('num_classes', 10, '')
flags.DEFINE_bool('from_npy', False, '')
flags.DEFINE_integer('patch_size', 64, '')

FLAGS = flags.FLAGS

def get_patches(img, max_patches):
    h = img.shape[0]
    w = img.shape[1]
    n = 0
    while n < max_patches:
        start_r = np.random.randint(0, h - FLAGS.patch_size, 1)[0]
        start_c = np.random.randint(0, w - FLAGS.patch_size, 1)[0]
        patch = img[start_r:start_r + FLAGS.patch_size, start_c:start_c + FLAGS.patch_size, :]
        n = n + 1
        yield patch

def main(_):

    # train_example_batch, train_label_batch = input_pipeline([FLAGS.data_dir + '/spc_train.tfrecords'], FLAGS.batch_size)
    # valid_example_batch, valid_label_batch = input_pipeline([FLAGS.data_dir + '/spc_valid.tfrecords'], FLAGS.batch_size)

    train_list = os.listdir('./tmp/train')
    valid_list = os.listdir('./tmp/valid')

    meta = {}
    f = open('spc_classes.txt')
    line = f.readline()
    while line:
        label, msg = line.split(' ')
        msg = msg.split('\n')[0]
        meta[msg] = int(label)
        line = f.readline()
    f.close()



    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 64, 64, 3], 'x')
        tf.summary.image('show', x, 1)

    with tf.name_scope('label'):
        y_ = tf.placeholder(tf.int64, [None, 1], 'y')
    with tf.variable_scope('net'):
        # with tf.device('/gpu:7'):
            y, keep_prob = res.build_net(x, 3, FLAGS.num_classes)

    pred = tf.argmax(y, 1)
    correct_prediction = tf.equal(tf.reshape(pred, [-1, 1]), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    with tf.Session() as sess:
        saver = tf.train.Saver(name="saver")

        if tf.gfile.Exists(os.path.join(FLAGS.ckpt_dir, 'checkpoint')):
            saver.restore(sess, os.path.join(FLAGS.ckpt_dir, 'model.ckpt'))
        else:
            sess.run(tf.global_variables_initializer())
        if FLAGS.from_npy:
            def feed_dict(train, kk=FLAGS.dropout):
                """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""

                def get_batch(root, mlist):

                    data = np.ndarray(shape=(FLAGS.batch_size, 64, 64, 3), dtype=np.uint8)
                    labels = np.ndarray(shape=(FLAGS.batch_size, 1), dtype=np.int64)
                    idx = np.random.randint(0, len(mlist), FLAGS.batch_size)
                    for i in xrange(FLAGS.batch_size):
                        dic = np.load(os.path.join(root, mlist[idx[i]])).item()
                        data[i, ...] = dic['patch']
                        labels[i, ...] = dic['label']
                    data = (data - 128.)/128.
                    labels.astype(np.int64)

                    return data, labels


                if train:
                    xs, ys = get_batch('./tmp/train', train_list)
                    k = kk
                else:
                    xs, ys = get_batch('./tmp/valid', valid_list)
                    k = 1.0
                return {x: xs, y_: ys}

            for i in range(FLAGS.start_step, FLAGS.max_steps + 1):
                pr, acc = sess.run([pred, accuracy], feed_dict=feed_dict(False))
                # test_writer.add_summary(summary, i)
                print i
                print acc
                print pr
                pr, acc = sess.run([pred, accuracy], feed_dict=feed_dict(True))
                # train_writer.add_summary(summary, i)
                print acc
                print pr
        else:
            classes = os.listdir(FLAGS.data_dir)
            for class_name in classes:
                img_names = os.listdir(os.path.join(FLAGS.data_dir, class_name))
                for img_name in img_names:
                    fullpath = os.path.join(FLAGS.data_dir, class_name, img_name)
                    img = plt.imread(fullpath)
                    data = np.ndarray(shape=(FLAGS.batch_size, 64, 64, 3), dtype=np.uint8)
                    i = 0
                    for patch in get_patches(img, FLAGS.batch_size):
                        data[i, :] = patch
                        i = i + 1
                    data = (data - 128.) / 128.
                    prediction = list(sess.run(pred, feed_dict={x: data}))
                    print prediction
                    count = np.ndarray(shape=(10, 1), dtype=np.int32)
                    for i in xrange(FLAGS.num_classes):
                        count[i] = prediction.count(i)
                    print 'predict %d while true label is %d' % (np.argmax(count, 0), meta[class_name])


if __name__ == '__main__':
    tf.app.run()

