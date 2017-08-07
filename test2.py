import tensorflow as tf
import tensorflow.contrib.slim as slim
import os, sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import cv2
import numpy as np
import model


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
image = tf.map_fn(lambda image: tf.image.per_image_standardization(image),
                   images, dtype=tf.float32)
prediction = model.lenet_advanced(image, num_classes, False, 1)

local_vars_init_op = tf.local_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(local_vars_init_op)
    saver.restore(sess, '/home/kris/PycharmProjects/traffic_sign_recognition/lenet_parameters_R.ckpt')

    cam = cv2.VideoCapture(1)
    while True:
        _, img = cam.read()
        img_copy = img.copy()
        img_copy = img.copy()
        img = cv2.resize(img, (64, 64))
        #img = tf.cast(img, tf.float32)
        #img = tf.image.per_image_standardization(img)
        #img = sess.run(tf.expand_dims(img, axis=0))

        pred, imgs = sess.run([prediction, image], feed_dict={images : [img]})
        print(imgs[0])
        res = [int(e) for e in np.argmax(pred, 1)]
        cv2.putText(img_copy, str(res[0]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 100, 0), 3)
        cv2.imshow("cam", img_copy)
        cv2.waitKey(0)
    sess.close()