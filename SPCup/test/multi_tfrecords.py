import tensorflow as tf
import matplotlib.pyplot as plt
import os
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
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True

image, label = input_pipeline(['/home/amax/QiYao/res/Res32_3/train/q85_train.tfrecords', '/home/amax/QiYao/res/Res32_3/train/q80_train.tfrecords'], 100)

with tf.Session(config = config) as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    im, lab = sess.run([image, label])
    print lab
    for i in range(100):
        plt.imshow(im[i, :])
        plt.show()
    coord.request_stop()
    coord.join(threads)
