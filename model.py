import tensorflow as tf
import tensorflow.contrib.slim as slim


def vgg_16(inputs, num_classes, is_training=True,
           dropout_keep_prob=0.5):

    with tf.variable_scope('vgg_16', reuse=None) as sc:
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.conv2d(net, 4096, [8, 8], padding='VALID', scope='fc3')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout3')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc4')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout4')
        net = slim.conv2d(net, num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='fc5')
        net = tf.squeeze(net, [1, 2], name='fc5/squeezed')
    return net


def lenet(images, num_classes, is_training=False,
          dropout_keep_prob=0.5,
          scope='LeNet'):

  with tf.variable_scope(scope, 'LeNet', [images, num_classes]):
    net = slim.conv2d(images, 32, [5, 5], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    net = slim.flatten(net)
    net = slim.fully_connected(net, 1024, scope='fc3')
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                       scope='dropout3')
    logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                  scope='fc4')

  return logits

def lenet_advanced(images, num_classes, is_training=False,
          dropout_keep_prob=0.5,
          scope='LeNet'):

  with tf.variable_scope(scope, 'LeNet_Advanced', [images, num_classes]):
    net = slim.conv2d(images, 32, [5, 5], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    net = slim.conv2d(net, 64, [5, 5], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool3')
    net = slim.flatten(net)
    net = slim.fully_connected(net, 1024, scope='fc4')
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                       scope='dropout4')
    logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                  scope='fc5')

  return logits