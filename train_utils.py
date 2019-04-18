"""
utils function for training and evaluation/metrics
"""
import os
import keras
import tensorflow as tf
import numpy as np
from keras.callbacks import TensorBoard


def launch_tensorboard():
    log_dir = 'D:\herschel\\navigation\keras_implementation\log'
    cmd = 'tensorboard --logdir={} --host=0.0.0.0 --port=8000'.format(log_dir)
    os.system(cmd)
    return


def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = np.argmax(y_pred_in, axis=-1)

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.array(np.mean(metric), dtype=np.float32)


def my_iou_metric(label, pred):
    metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float32)
    return metric_value


def create_callbacks(model_name):
    callback = []
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(
            '',
            model_name
        ),
        verbose=1
    )
    log_dir = './log'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    tensorboard = TensorBoard(log_dir=os.path.join(log_dir, model_name))
    callback.append(checkpoint)
    callback.append(tensorboard)
    return callback


def valid_acc_metric(y_true, y_pred):
    """
    compute acc on valid samples
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true = keras.backend.reshape(y_true, (-1,))
    y_pred = keras.backend.argmax(y_pred, axis=-1)
    y_pred = keras.backend.reshape(y_pred, (-1,))
    valid_indices = tf.to_int32(keras.backend.less(y_true, 255))
    y_true = tf.dynamic_partition(y_true, valid_indices, num_partitions=2)[1]
    y_pred = tf.dynamic_partition(y_pred, valid_indices, num_partitions=2)[1]
    y_true = keras.backend.cast(y_true, tf.int16)
    y_pred = keras.backend.cast(y_pred, tf.int16)
    acc = keras.backend.mean(keras.backend.equal(y_true, y_pred))
    return acc


