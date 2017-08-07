import numpy as np
import sys, os
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import cv2
import tensorflow as tf
from read_data import load_data_from_files

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

tfrecords_filename = 'CTIOne_Traffic_Sign.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

data_dir = "CTIOne_Dataset/Training"

directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]

for d in directories:
    images = []
    label_dir = os.path.join(data_dir, d)
    file_names = [os.path.join(label_dir, f)
                  for f in os.listdir(label_dir)
                  if f.endswith(".ppm") or f.endswith(".JPG")
                  or f.endswith(".jpg") or f.endswith(".png")]

    # Augment the dataset with rotation and blurring
    for f in file_names:
        image = cv2.imread(f)
        label = int(d)
        if image is not None:

            height = 64
            width = 64

            image = cv2.resize(image, (height, width))

            image_raw = image.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'image_raw': _bytes_feature(image_raw),
                'mask_raw': _int64_feature(label)}))

            writer.write(example.SerializeToString())

writer.close()