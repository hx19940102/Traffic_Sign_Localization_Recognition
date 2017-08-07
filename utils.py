import tensorflow as tf
import numpy as np


def cross_entropy_loss(preds, labels):
    e = 10e-6
    softmax_pred = tf.nn.softmax(preds)
    loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(softmax_pred + e), 1) -
                          tf.reduce_sum((1 - labels) * tf.log(1 - softmax_pred + e), 1))
    return loss

def calculate_accuracy(preds, labels):
    pred_class = np.argmax(preds, 1)
    index = [i for i in range(0, len(labels)) if pred_class[i] == labels[i]]
    return len(index) / float(preds.shape[0])