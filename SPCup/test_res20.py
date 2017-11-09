import tensorflow as tf
import tensorflow.contrib.losses as loss
import tensorflow.contrib.layers as layers
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import time

import resnet as res

flags = tf.app.flags

# flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
# flags.DEFINE_string('data_dir', '../patches', 'data direction')
# flags.DEFINE_string('log_dir', './logs', 'log direction')
flags.DEFINE_string('ckpt_dir', './res20/ckpt', 'check point direction')
flags.DEFINE_string('data_dir', '../data', '')
flags.DEFINE_string('meta_path', './spc_classes.txt', '')
flags.DEFINE_integer('patch_size', 64, '')
FLAGS = flags.FLAGS


def main(_):
    meta = {}
    f = open(FLAGS.meta_path, 'rb')

    line = f.readline()
    while line:
        a = line.split(' ')
        b = a[1].split('\n')
        meta[int(a[0])] = b[0]
        line = f.readline()
    f.close()



    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 64, 64, 3], 'x')
        # tf.summary.image('show', x, 1)

    with tf.variable_scope('net'):
        y, keep_prob = res.build_net(x, 3, 10)

    pred = tf.argmax(y, 1)

    with tf.name_scope("saver"):
        saver = tf.train.Saver(name="saver")

    with tf.Session() as sess:
        if tf.gfile.Exists(os.path.join(FLAGS.ckpt_dir, 'checkpoint')):
            saver.restore(sess, os.path.join(FLAGS.ckpt_dir, 'model.ckpt'))

        data = np.ndarray(shape=(FLAGS.batch_size, 64, 64, 3), dtype=np.uint8)
        labels = np.ndarray(shape=(FLAGS.batch_size, 1), dtype=np.int64)
        t = time.time()
        for i in xrange(99):
            dic = np.load('tmp/train/%d.npy' % (i+1)).item()
            data[i, ...] = dic['patch']
            labels[i, ...] = dic['label']

        print time.time() - t

        # for i in xrange(100):
        #     plt.imshow(patches[i, :])
        #     plt.show()
        # patches = (patches - 128.)/128.
        # print sess.run(pred, feed_dict={x: patches})
        # valid = os.listdir('./tmp/valid')
        # for file in valid:
        #     full_path = os.path.join('./tmp/valid', file)
        #     dic = np.load(full_path).item()
        #     patch = (dic['patch'] - 128.)/128.
        #     patch = patch.reshape((1, 64, 64, 3)).astype(np.float32)
        #     label = dic['label']
        #     prediction = sess.run(pred, feed_dict={x: patch})
        #     print 'label: %d, prediction: %d' % (label, prediction)

        # classes = os.listdir(FLAGS.data_dir)
        #
        # for label, class_name in enumerate(classes):
        #     img_names = os.listdir(os.path.join(FLAGS.data_dir, class_name))
        #     for img_name in img_names:
        #         full_path = os.path.join(FLAGS.data_dir, class_name, img_name)
        #         img = plt.imread(full_path)
        #         h = img.shape[0]
        #         w = img.shape[1]
        #         for i in xrange(10):
        #             start_r = np.random.randint(0, h - FLAGS.patch_size, 1)[0]
        #             start_c = np.random.randint(0, w - FLAGS.patch_size, 1)[0]
        #             patch = img[start_r:start_r + FLAGS.patch_size, start_c:start_c + FLAGS.patch_size, :]
        #             patch = patch.reshape((1, 64, 64, 3))
        #             stddev = np.std(patch, ddof=1)
        #             adjusted_stddev = stddev if stddev > 1.0 / FLAGS.patch_size else 1.0 / FLAGS.patch_size
        #             patch = (patch - np.mean(patch)) / adjusted_stddev
        #
        #             result = sess.run(pred ,feed_dict={x: patch})
        #             print img_name + ' captured by ' + class_name + ', patch ' + str(i) + ' prediction: ' + \
        #                 meta[result[0]]




if __name__ == '__main__':
    tf.app.run()

