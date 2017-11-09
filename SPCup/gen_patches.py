# coding:utf-8
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from random import shuffle
import numpy as np


flags = tf.app.flags

flags.DEFINE_string('data_dir', '../data', 'Data direction')
flags.DEFINE_string('out_dir', '../patches', 'Output direction')
flags.DEFINE_integer('patch_size', 64, '')
flags.DEFINE_integer('max_patches', 100, 'number of patches that one image can generate at most')
flags.DEFINE_bool('keep', True, '')
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

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(_):
    if not tf.gfile.Exists(FLAGS.out_dir):
        tf.gfile.MakeDirs(FLAGS.out_dir)
    tf.gfile.MakeDirs('./tmp/train')
    tf.gfile.MakeDirs('./tmp/valid')
    classes = os.listdir(FLAGS.data_dir)

    # train_name = os.path.join(FLAGS.out_dir, 'spc_train.tfrecords')
    # valid_name = os.path.join(FLAGS.out_dir, 'spc_valid.tfrecords')

    # train_writer = tf.python_io.TFRecordWriter(train_name)
    # valid_writer = tf.python_io.TFRecordWriter(valid_name)

    spc_classes = open('spc_classes.txt', 'w')
    n = 0
    for label, class_name in enumerate(classes):
        spc_classes.write(('%d ' % label) + class_name + '\n')

        img_names = os.listdir(os.path.join(FLAGS.data_dir, class_name))
        for img_name in img_names:
            full_path = os.path.join(FLAGS.data_dir, class_name, img_name)
            print 'processing ' + full_path
            img = plt.imread(full_path)
            dice = np.random.randint(0, 5, 1)
            # writer = train_writer if dice != 0 else valid_writer
            set = 'train' if dice != 0 else 'valid'
            for n, patch in enumerate(get_patches(img, FLAGS.max_patches)):
                np.save(os.path.join('./tmp', set, class_name + '_' + img_name + '_' + str(n) + '.npy'), {'label': label, 'patch': patch})

    # train = os.listdir('./tmp/train')
    # valid = os.listdir('./tmp/valid')
    # idx = range(len(train))
    # shuffle(idx)
    # for i in idx:
    #     dic = np.load(os.path.join('./tmp/train', train[i])).item()
    #     patch = dic['patch']
    #     label = dic['label']
    #
    #     patch_raw = patch.tostring()
    #     label_raw = np.array([label]).astype(np.int32).tostring()
    #     example = tf.train.Example(features=tf.train.Features(feature={
    #         'patch_raw': _bytes_feature(patch_raw),
    #         'label': _bytes_feature(label_raw)}))
    #     train_writer.write(example.SerializeToString())
    #
    # idx = range(len(valid))
    # shuffle(idx)
    # for i in idx:
    #     dic = np.load(os.path.join('./tmp/valid', valid[i])).item()
    #     patch = dic['patch']
    #     label = dic['label']
    #
    #     patch_raw = patch.tostring()
    #     label_raw = np.array([label]).astype(np.int32).tostring()
    #     example = tf.train.Example(features=tf.train.Features(feature={
    #         'patch_raw': _bytes_feature(patch_raw),
    #         'label': _bytes_feature(label_raw)}))
    #     valid_writer.write(example.SerializeToString())
    #
    # train_writer.close()
    # valid_writer.close()
    spc_classes.close()
    if not FLAGS.keep:
        tf.gfile.DeleteRecursively('./tmp')


if __name__ == "__main__":
    tf.app.run()