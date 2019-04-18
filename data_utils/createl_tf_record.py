"""Converts PASCAL dataset to TFRecords file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import PIL.Image
import tensorflow as tf
from data_utils import dataset_util


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def dict_to_tf_example(image_path,
                       label_path):
  """Convert a single image and label to tf.Example proto.
     one image path with two corresponding labels(classification label\segmentation label)

  Args:
    image_path: Path to a single PASCAL image.
    label_path: its corresponding labels, a list of two[int, label_path].

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by image_path is not a valid JPEG or
                if the label pointed to by label_path is not a valid PNG or
                if the size of image does not match with that of label.
  """
  # image data
  with tf.gfile.GFile(image_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  classification_label, segmentation_label_dir = label_path
  # segmentation label
  if segmentation_label_dir != 0:
      with tf.gfile.GFile(segmentation_label_dir, 'rb') as fid:
        encoded_label = fid.read()
      encoded_label_io = io.BytesIO(encoded_label)
      segmentation_label = PIL.Image.open(encoded_label_io)
      pass
  else:
      segmentation_label = PIL.Image.new('L', image.size, (255, ))
      segmentation_label.save("test.png", 'png')
      # convert into bytes
      with tf.gfile.GFile('test.png', 'rb') as fid:
        encoded_label = fid.read()
  if image.size != segmentation_label.size:
    raise ValueError('The size of image does not match with that of label.')
  width, height = image.size
  example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': int64_feature(height),
    'image/width': int64_feature(width),
    'image/encoded': bytes_feature(encoded_jpg),
    'image/format': bytes_feature('png'.encode('utf8')),
    'label/cl_label': int64_feature(classification_label),
    'label/seg_label': bytes_feature(encoded_label),
    'label/format': bytes_feature('png'.encode('utf8')),
  }))
  return example


def create_tf_record(output_filename,
                     image_dir,
                     label_dir,
                     ):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    image_dir:  a list of images dir.
    label_dir: a list of files dir.
    examples: Examples to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, (image_path, label_path) in enumerate(zip(image_dir, label_dir)):
    if idx % 200 == 0:
      print ('On image {} of {}'.format(idx, len(image_dir)))
    if not os.path.exists(image_path):
      tf.logging.warning('Could not find %s, ignoring example.', image_path)
      continue
    elif not os.path.exists(label_path):
      tf.logging.warning('Could not find %s, ignoring example.', label_path)
      continue
    if image_path.split('\\')[-1].split('_left')[0] != label_path.split('\\')[-1].split('_gt')[0]:
        tf.logging.warning('img and label not match', image_path.split('\\')[-1].split('_left')[0], label_path.split('\\')[-1].split('_gt')[0])
        continue
    try:
      tf_example = dict_to_tf_example(image_path, label_path)
      writer.write(tf_example.SerializeToString())
    except ValueError:
      tf.logging.warning('Invalid example: %s, ignoring.')
  print ('final sample:{}'.format(len(image_dir)))
  writer.close()


def create_tf_record_test(output_filename,
                     image_dir,
                     label_dir,
                     ):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    image_dir:  a list of images dir.
    label_dir: a list of files dir.
    examples: Examples to parse and save to tf record.
  """
  class_labels, seg_label_dir = label_dir
  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, (image_path, cl_label, label_path) in enumerate(zip(image_dir, class_labels, seg_label_dir)):
    if idx % 200 == 0:
      print ('On image {} of {}'.format(idx, len(image_dir)))
    if not os.path.exists(image_path):
      tf.logging.warning('Could not find %s, ignoring example.', image_path)
      continue
    if label_path != 0:
        if not os.path.exists(label_path):
          tf.logging.warning('Could not find %s, ignoring example.', label_path)
          continue
        if image_path.split('\\')[-1].split('_left')[0] != label_path.split('\\')[-1].split('_gt')[0]:
            tf.logging.warning('img and label not match', image_path.split('\\')[-1].split('_left')[0], label_path.split('\\')[-1].split('_gt')[0])
            continue
    try:
      tf_example = dict_to_tf_example(image_path, [cl_label, label_path])
      writer.write(tf_example.SerializeToString())
    except ValueError:
      tf.logging.warning('Invalid example: %s, ignoring.')
  print ('final sample:{}'.format(len(image_dir)))
  writer.close()


def main(arg):
    """
    prepare dataset for two branches: classification and segmentation
    for classification dataset: it's segmentation label_dir is set to 0(invalid path)
    for segmentation dataset: it's classification label is set to -1(invalid class)
    :param arg:
    :return:
    """
    tf.logging.info("Reading from dataset")
    # prepare file path and labels
    # segmentation branch
    train_img = dataset_util.get_file_list('D:\herschel\\navigation\data\leftImg8bit_trainvaltest\leftImg8bit\\train')
    train_labels = dataset_util.read_examples_list('D:\herschel\\navigation\data\gtFine_trainvaltest\gtFine\\train')
    # label as -1 if no corresponding labels
    seg_cl_label_train = [-1] * len(train_labels)
    val_labels = dataset_util.read_examples_list('D:\herschel\\navigation\data\gtFine_trainvaltest\gtFine\\val')
    val_img = dataset_util.get_file_list('D:\herschel\\navigation\data\leftImg8bit_trainvaltest\leftImg8bit\\val')
    seg_cl_label_val = [-1] * len(val_labels)
    # classification branch
    cl_data_train, cl_labels_train = dataset_util.read_classification_data('D:\herschel\\navigation\data\classification\\train')
    cl_seg_train = [0] * len(cl_labels_train)
    cl_data_val, cl_labels_val = dataset_util.read_classification_data('D:\herschel\\navigation\data\classification\\validation')
    cl_seg_val = [0] * len(cl_labels_val)
    # merge classification and segmentation labels
    # train_data = train_img + cl_data_train
    # train_labels = [seg_cl_label_train + cl_labels_train, train_labels + cl_seg_train]
    # val_data = val_img + cl_data_val
    # val_labels = [seg_cl_label_val + cl_labels_val, val_labels + cl_seg_val]
    # generate seperate tfrecord for classification data
    train_data = cl_data_train
    train_labels = [cl_labels_train, cl_seg_train]
    val_data = cl_data_val
    val_labels = [cl_labels_val, cl_seg_val]
    # output path
    prefix = 'D:\herschel\\navigation\\tf_records'
    train_output_path = os.path.join(prefix, 'cl_train.record')
    val_output_path = os.path.join(prefix, 'cl_val.record')

    create_tf_record_test(train_output_path, train_data, train_labels)
    create_tf_record_test(val_output_path, val_data, val_labels)


if __name__ == '__main__':
  tf.app.run()


