import os, sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import cv2
import tensorflow as tf


def load_data_from_files(data_dir):
    """Loads a data set and returns two lists:

    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".ppm") or f.endswith(".JPG")
                      or f.endswith(".jpg")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(cv2.imread(f))
            labels.append(int(d))
    return images, labels

def load_data_from_tfrecords(filename_queue, batch_size):

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.int64)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    label = tf.cast(features['mask_raw'], tf.int32)

    image_shape = tf.pack([height, width, 3])
    image = tf.reshape(image, image_shape)

    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                           target_height=64,
                                                           target_width=64)


    images, labels = tf.train.shuffle_batch([resized_image, label],
                                            batch_size=batch_size,
                                            capacity=20000,
                                            num_threads=2,
                                            min_after_dequeue=20 * batch_size,
                                            allow_smaller_final_batch=True)

    return images, labels


