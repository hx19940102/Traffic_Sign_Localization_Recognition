import tensorflow as tf
import tensorflow.contrib.slim as slim
import os, sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import cv2
import numpy as np
import model
from localize import traffic_sign_locate


CLASSIFIER = cv2.CascadeClassifier("lbpCascade.xml")
downscale = 2
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
num_classes = 5

sign_map = {}
with open('signnames.csv', 'r') as f:
	for line in f:
		line = line[:-1]  # strip newline at the end
		sign_id, sign_name = line.split(',')
		sign_map[int(sign_id)] = sign_name


images = tf.placeholder(tf.uint8, [None, 64, 64, 3])
images = tf.cast(images, tf.float32)
image_batch = tf.map_fn(lambda image: tf.image.per_image_standardization(image),
                        images, dtype=tf.float32)
prediction = model.lenet_advanced(image_batch, num_classes, False, 1)

local_vars_init_op = tf.local_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(local_vars_init_op)
    saver.restore(sess, '/home/kris/PycharmProjects/traffic_sign_recognition/lenet_parameters.ckpt')

    cam = cv2.VideoCapture(1)
    while True:
        _, img = cam.read()

        locations = traffic_sign_locate(img, CLASSIFIER, downscale)
        if len(locations) is not 0:
            signs = [cv2.resize(img[location[1]:location[1] + location[3],
                     location[0]:location[0] + location[2]], (64, 64))
                     for location in locations]

            pred = sess.run(prediction, feed_dict={images: signs})

            res = [int(e) for e in np.argmax(pred, 1)]
            sign_name = [sign_map[i] for i in res]
            for i in range(len(locations)):
                cv2.rectangle(img, (locations[i][0], locations[i][1]),
                              (locations[i][0] + locations[i][2], locations[i][1] + locations[i][3]),
                              (0, 0, 255), 3)
                cv2.putText(img, sign_name[i], (locations[i][0], locations[i][1]), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 100, 0), 3)
        cv2.imshow("cam", img)
        cv2.waitKey(10)
    sess.close()