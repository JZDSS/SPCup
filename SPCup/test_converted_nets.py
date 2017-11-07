# coding:utf-8
import tensorflow as tf
import cv2

from ResNet50 import ResNet_50 as res50
# from ResNet101 import ResNet_101 as res101
# from ResNet152 import ResNet_152 as res152
#
#
# flags = tf.app.flags
#
# flags.DEFINE_string('logdir', './logs', 'Log direction')
# flags.DEFINE_string('ckpt', './ckpt', 'Check point direction')
# flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
# flags.DEFINE_integer('batch_size', 100, 'Batch size')
# flags.DEFINE_integer('num_steps', 100000, 'Number of steps to run')
#
# FLAGS = flags.FLAGS


def main(_):
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    net = res50({'data': images})
    prob = net.layers['prob']
    pred = tf.arg_max(prob, 1)
    img = cv2.imread('dog.jpg')
    img = img.reshape((1, 224, 224, 3))

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./logs", sess.graph)
        writer.flush()
        writer.close()
        # sess.run(tf.initialize_all_variables())
        net.load('../tfmodels/ResNet50.npy', sess)

        print(sess.run(pred, feed_dict={images: img}))



if __name__ == "__main__":
    tf.app.run()