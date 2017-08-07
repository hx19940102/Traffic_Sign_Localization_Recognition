from read_data import load_data_from_tfrecords
import tensorflow as tf
import tensorflow.contrib.slim as slim
import model, utils
import os, sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import cv2
import numpy as np


training_data_dir = "CTIOne_Traffic_Sign.tfrecords"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
num_classes = 5
batch_size = 256

filename_queue = tf.train.string_input_producer(
    [training_data_dir], num_epochs=None)

image_batch, label_batch = load_data_from_tfrecords(filename_queue,
                                                    batch_size)

label = tf.one_hot(label_batch, num_classes, 1, 0)
label = tf.reshape(tf.cast(label, tf.float32), [batch_size, num_classes])
image = tf.cast(image_batch, tf.float32)
image = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
                  image, dtype=tf.float32)


output = model.lenet_advanced(image, num_classes, True, 0.5)
output = tf.reshape(tf.cast(output, tf.float32), [batch_size, num_classes])

loss = utils.cross_entropy_loss(output, label)

train = tf.train.AdamOptimizer(0.001).minimize(loss)

global_vars_init_op = tf.global_variables_initializer()
local_vars_init_op = tf.local_variables_initializer()
combined_op = tf.group(local_vars_init_op, global_vars_init_op)
model_variables = slim.get_model_variables()
saver = tf.train.Saver(model_variables)

with tf.Session() as sess:
    sess.run(combined_op)
    # saver.restore(sess, '/home/kris/PycharmProjects/traffic_sign_recognition/lenet_parameters.ckpt')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(50000):
        ignore, pred, error, images, truth = sess.run([train, output, loss, image_batch, label_batch])
        #for i in range(len(images)):
        #    cv2.imshow("image" + str(i), cv2.resize(images[i], (256, 256)));
        #    print(truth[i])
        #cv2.waitKey(0)
        accuracy = utils.calculate_accuracy(pred, truth)
        print("%d round loss = %f, accuracy = %f" % (i, error, accuracy))
        if i % 199 == 0:
            saver.save(sess, '/home/kris/PycharmProjects/traffic_sign_recognition/lenet_parameters.ckpt')
    coord.request_stop()
    coord.join(threads)

saver.save(sess, '/home/kris/PycharmProjects/traffic_sign_recognition/lenet_parameters.ckpt')
