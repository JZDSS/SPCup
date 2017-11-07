import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

    image = tf.reshape(image, [64, 64, 3])
    ground_truth = tf.reshape(ground_truth, [1])
    return image, ground_truth

image, label = read_from_tfrecord(tf.train.string_input_producer(['../patches/spc_train.tfrecords']))
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in xrange(10):
        im, lab = sess.run([image, label])
        print(lab)
        plt.imshow(im)
        plt.show()
    coord.request_stop()
    coord.join(threads)