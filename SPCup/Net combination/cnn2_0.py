#coding:utf-8
import tensorflow as tf
import os
import time
import my_net as net
traindata = ['/home/wenlive/文档/mycnn/traindata_Lab_Ima.bin']
testdata = ['/home/wenlive/文档/mycnn/testdata_Lab_Ima.bin']


def get_single(filename_queue):
    dbytes = 794
    reader = tf.FixedLengthRecordReader(record_bytes=dbytes)
    _,value = reader.read(filename_queue)
    record = tf.decode_raw (value, tf.uint8)
    label = tf.reshape(tf.strided_slice (record,[0],[10],[1]),[1,10])
    label = tf.cast (label,tf.float32)
    image = tf.reshape(tf.strided_slice(record,[10],[dbytes],[1]),[1,784])
    image = tf.cast (image,tf.float32)
    return label,image
def get_batch(filename,batch_size):
    filename_queue = tf.train.string_input_producer(filename,num_epochs=None)
    label,image = get_single(filename_queue)
    min_after_dequeue = 5*batch_size
    capacity = 10*batch_size
    labels,images = tf.train.shuffle_batch(
                       [label,image],
                        batch_size=batch_size,
                        capacity=capacity,
                        min_after_dequeue=min_after_dequeue)
    return tf.reshape(labels,[batch_size,10]),tf.reshape(images,[batch_size,784])

def main(_):
	#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
	config = tf.ConfigProto()
	#config.gpu_options.per_process_gpu_memory_fraction = 0.9
	#config.gpu_options.allow_growth = True
	x = tf.placeholder(tf.float32, [None, 784])
	y_ = tf.placeholder(tf.float32, [None, 10])
	phase = tf.placeholder(tf.bool,name='phase')

	y_conv = net.deepnn(x,phase)

	with tf.name_scope('loss'):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
		loss = tf.reduce_mean(cross_entropy)
		tf.summary.scalar('loss',loss,collections=None)
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	global_step = tf.Variable(0,trainable=False)
	learning_rate = tf.train.exponential_decay(0.001,global_step,1000,0.9,staircase=True)
	with tf.control_dependencies(update_ops):
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step)
	with tf.name_scope('accuracy'):
		correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
		correct_prediction = tf.cast(correct_prediction, tf.float32)
		accuracy = tf.reduce_mean(correct_prediction)
		tf.summary.scalar('accuracy',accuracy,collections=None)

	trianlabels,trainimages = get_batch(traindata,batch_size=128)
	testlabels,testimages = get_batch(testdata,batch_size=500)
	with tf.name_scope("saver"):
		saver = tf.train.Saver(name="saver")
	with tf.Session(config = config) as sess:
		merged_summary_op = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter('/home/wenlive/文档/mycnn/cnn2_0_log',sess.graph)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		sess.run(tf.global_variables_initializer())
		for i in range(2000):
			batch_y,batch_x= sess.run([trianlabels,trainimages])
			# if (i+1) % 100 == 0:
			# 	train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y,phase:False})
			# 	testy,testx = sess.run([testlabels,testimages])
			# 	test_accuracy = accuracy.eval(feed_dict={x: testx, y_: testy,phase:False})
			# 	print time.asctime(time.localtime())
			# 	print('step %d, train accuracy %g, test accuracy %g' %(i+1, train_accuracy,test_accuracy))
			# 	summary_str = sess.run(merged_summary_op,
             #                       feed_dict={x: testx,y_:testy,phase:False})
			# 	summary_writer.add_summary(summary_str,i)
			saver.save(sess, os.path.join('./ckpt3', 'net3'))
			train_step.run(feed_dict={x: batch_x, y_: batch_y,phase:True})
		coord.request_stop()
		coord.join(threads)
if __name__ == '__main__':
	tf.app.run()

