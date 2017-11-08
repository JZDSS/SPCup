# coding:utf-8
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses

from ResNet50 import ResNet_50 as res50
from ResNet101 import ResNet_101 as res101
from ResNet152 import ResNet_152 as res152


flags = tf.app.flags

flags.DEFINE_string('log_dir', './logs', 'Log direction')
flags.DEFINE_string('ckpt_dir', './ckpt', 'Check point direction')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
flags.DEFINE_integer('batch_size', 100, 'Batch size')
flags.DEFINE_integer('num_steps', 100000, 'Number of steps to run')
flags.DEFINE_integer('decay_steps', 100, '')
flags.DEFINE_float('decay_rate', 0.98, '')
flags.DEFINE_float('momentum', 0.95, '')
flags.DEFINE_integer('start_step', 0, '')
flags.DEFINE_integer('max_steps', 10000, '')

FLAGS = flags.FLAGS

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




def main(_):

    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)
    train_example_batch, train_label_batch = input_pipeline(['../patches/spc_train.tfrecords'], FLAGS.batch_size)

    tf.summary.image('show', train_example_batch, 1)
        # with tf.name_scope('input'):
        #     x = tf.placeholder(tf.float32, [None, 64, 64, 3], 'x')
        #     tf.summary.image('show', x, 1)
        #
        # with tf.name_scope('label'):
        #     y_ = tf.placeholder(tf.int32, [None, 1], 'y')
    with tf.device('/gpu:7'):
        net = res50({'data': train_example_batch})

        fc10 = net.layers['fc10']
    with tf.name_scope('scores'):
        with tf.device('/gpu:7'):
            losses.sparse_softmax_cross_entropy(fc10, train_label_batch, scope='cross_entropy')
            total_loss = tf.contrib.losses.get_total_loss(add_regularization_losses=True, name='total_loss')

            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.cast(tf.reshape(tf.argmax(fc10, 1), [-1, 1]), tf.int32),
                                              train_label_batch)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('train'):
        global_step = tf.Variable(FLAGS.start_step, name="global_step")
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
            global_step, FLAGS.decay_steps, FLAGS.decay_rate, True, "learning_rate")
        # learning_rate = tf.train.piecewise_constant(global_step, [32000, 48000], [0.1, 0.01, 0.001])
        with tf.device('/gpu:7'):
            train_step = tf.train.MomentumOptimizer(learning_rate, momentum=FLAGS.momentum).minimize(
                total_loss, global_step=global_step)
    tf.summary.scalar('lr', learning_rate)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        writer.flush()
        tf.global_variables_initializer().run()
        net.load('../tfmodels/ResNet50.npy', sess, ignore_missing=True)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # print sess.run(fc10)

        for i in xrange(FLAGS.start_step, FLAGS.max_steps + 1):
            # if i % 1000 == 0 and i != 1:
            #     time.sleep(60)
            if i % 100 == 0 and i != 0:  # Record summaries and test-set accuracy
                acc, summary = sess.run([accuracy, merged])
                writer.add_summary(summary, i)
                print acc
            sess.run(train_step)


        coord.request_stop()
        coord.join(threads)
        writer.close()



if __name__ == "__main__":
    tf.app.run()