"""
utils functions for data set preparation
"""
import os

import tensorflow as tf
from config_old import ModelConfig
from data_utils import preprocessing
from data_utils.preprocessing import mean_image_subtraction


def read_classification_data(data_dir, interested_categories=None):
    """
    read classification data from given folder and output a list of [[image_dir], [corresponding labels]]
    :param data_dir: dataset folder
    :return: a tupe of (data_set, label)
    """
    if interested_categories is None:
        interested_categories = ModelConfig.classification_categories
    file_list = get_file_list(data_dir)
    data_set = []
    label = []
    for index, item in enumerate(interested_categories):
        cate = list(filter(lambda x: item in x, file_list))
        cate_label = [index]*len(cate)
        data_set += cate
        label += cate_label
    return data_set, label


def get_file_list(folder_dir):
    """
    iteratively get file list under folder_dir
    :param folder_dir: folder
    :return: a list of files
    """
    file_list = []
    for root, dirs, files in os.walk(folder_dir, topdown=False):
        for name in files:
            sub_dir = os.path.join(root, name)
            if os.path.isfile(sub_dir):
                file_list.append(sub_dir)
        for name in dirs:
            sub_dir = os.path.join(root, name)
            if os.path.isfile(sub_dir):
                file_list.append(sub_dir)
    return file_list


def read_examples_list(path):
    """Read list of training or validation examples from labels folder.
    Args:
    path: absolute path to train/valid/ labels examples list file.
    Returns:
    list of example identifiers (strings).
    """
    # get files
    file_list = get_file_list(path)
    # filter out undesired file
    file_list = list(filter(lambda x: 'label' in x, file_list))
    return file_list


def make_initializable_iterator(dataset):
  """Creates an iterator, and initializes tables.

  This is useful in cases where make_one_shot_iterator wouldn't work because
  the graph contains a hash table that needs to be initialized.

  Args:
    dataset: A `tf.data.Dataset` object.

  Returns:
    A `tf.data.Iterator`.
  """
  iterator = dataset.make_initializable_iterator()
  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
  return iterator


def read_dataset(
    file_read_func, decode_func, input_files, config, num_workers=1,
    worker_index=0):
  """Reads a dataset, and handles repetition and shuffling.

  Args:
    file_read_func: Function to use in tf.data.Dataset.interleave, to read
      every individual file into a tf.data.Dataset.
    decode_func: Function to apply to all records.
    input_files: A list of file paths to read.
    config: A input_reader_builder.InputReader object.
    num_workers: Number of workers / shards.
    worker_index: Id for the current worker.

  Returns:
    A tf.data.Dataset based on config.
  """
  # Shard, shuffle, and read files.
  filenames = tf.concat([tf.matching_files(pattern) for pattern in input_files],
                        0)
  dataset = tf.data.Dataset.from_tensor_slices(filenames)
  dataset = dataset.shard(num_workers, worker_index)
  dataset = dataset.repeat(config.num_epochs or None)
  if config.shuffle:
    dataset = dataset.shuffle(config.filenames_shuffle_buffer_size,
                              reshuffle_each_iteration=True)

  # Read file records and shuffle them.
  # If cycle_length is larger than the number of files, more than one reader
  # will be assigned to the same file, leading to repetition.
  cycle_length = tf.cast(
      tf.minimum(config.num_readers, tf.size(filenames)), tf.int64)
  # TODO: find the optimal block_length.
  dataset = dataset.interleave(
      file_read_func, cycle_length=cycle_length, block_length=1)

  if config.shuffle:
    dataset = dataset.shuffle(config.shuffle_buffer_size,
                              reshuffle_each_iteration=True)

  dataset = dataset.map(decode_func, num_parallel_calls=config.num_readers)
  return dataset.prefetch(config.prefetch_buffer_size)


def get_filenames(is_training, data_dir):
  """Return a list of filenames.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: path to the the directory containing the input data.

  Returns:
    A list of file names.
  """
  if is_training:
    return [ModelConfig.train_tfrecord_dir]
  else:
    return [ModelConfig.val_tfrecord_dir]


def parse_record(raw_record):
    """Parse PASCAL image and label from a tf record."""
    keys_to_features = {
      'image/height':
      tf.FixedLenFeature((), tf.int64),
      'image/width':
      tf.FixedLenFeature((), tf.int64),
      'image/encoded':
      tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
      tf.FixedLenFeature((), tf.string, default_value='png'),
      'label/cl_label':
      tf.FixedLenFeature((), tf.int64),
      'label/seg_label':
      tf.FixedLenFeature((), tf.string, default_value=''),
      'label/format':
      tf.FixedLenFeature((), tf.string, default_value='png'),
    }

    parsed = tf.parse_single_example(raw_record, keys_to_features)

    # height = tf.cast(parsed['image/height'], tf.int32)
    # width = tf.cast(parsed['image/width'], tf.int32)

    image = tf.image.decode_image(
      tf.reshape(parsed['image/encoded'], shape=[]), 3)
    image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    image.set_shape([None, None, 3])
    #  segmentation labels
    label = tf.image.decode_image(
      tf.reshape(parsed['label/seg_label'], shape=[]), 1)
    label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
    label.set_shape([None, None, 1])
    # classification label
    cl_class = parsed['label/cl_label']
    return image, cl_class, label


def preprocess_image(image, cl_class, label, is_training):
    """Preprocess a single image of layout [height, width, depth]."""
    def _preprocess_for_seg(image, label):
        # Randomly scale the image and label.
        # image, label = preprocessing.random_rescale_image_and_label(
        #     image, label, ModelConfig.min_scale, ModelConfig.max_scale)
        # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
        image, label = preprocessing.random_crop_or_pad_image_and_label(
            image, label, ModelConfig.height, ModelConfig.width, None)
        # Randomly flip the image and label horizontally.
        image, label = preprocessing.random_flip_left_right_image_and_label(
            image, label)
        label = preprocessing.categories_selection(label, ModelConfig.interest_label)
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.int32)
        return image, label
    if is_training:
        # flag for data type: 1 for classification, 0 for semantic segmentation
        flag = tf.equal(cl_class, -1)
        # different process method for classification/segmentation data
        image, label = tf.cond(flag, lambda: _preprocess_for_seg(image, label),
                    lambda: preprocessing.resize_or_padding_image(image, label, ModelConfig.height, ModelConfig.width))

    # select interested labels
    image.set_shape([ModelConfig.height, ModelConfig.width, 3])
    label.set_shape([ModelConfig.height, ModelConfig.width, 1])
    return image, cl_class, label


def eval_input_fn(image_filenames, label_filenames=None, batch_size=1):
  """An input function for evaluation and inference.

  Args:
    image_filenames: The file names for the inferred images.
    label_filenames: The file names for the grand truth labels.
    batch_size: The number of samples per batch. Need to be 1
        for the images of different sizes.

  Returns:
    A tuple of images and labels.
  """
  # Reads an image from a file, decodes it into a dense tensor
  def _parse_function(filename, is_label):
    if not is_label:
      image_filename, label_filename = filename, None
    else:
      image_filename, label_filename = filename

    image_string = tf.read_file(image_filename)
    image = tf.image.decode_image(image_string)
    image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    image.set_shape([None, None, 3])

    image = mean_image_subtraction(image)

    if not is_label:
      return image
    else:
      label_string = tf.read_file(label_filename)
      label = tf.image.decode_image(label_string)
      label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
      label.set_shape([None, None, 1])

      return image, label

  if label_filenames is None:
    input_filenames = image_filenames
  else:
    input_filenames = (image_filenames, label_filenames)

  dataset = tf.data.Dataset.from_tensor_slices(input_filenames)
  if label_filenames is None:
    dataset = dataset.map(lambda x: _parse_function(x, False))
  else:
    dataset = dataset.map(lambda x, y: _parse_function((x, y), True))
  dataset = dataset.prefetch(batch_size)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()

  if label_filenames is None:
    images = iterator.get_next()
    labels = None
  else:
    images, labels = iterator.get_next()

  return images, labels


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
    """Input_fn using the tf.data input pipeline.

    Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

    Returns:
    A tuple of images and labels.
    """
    dataset = tf.data.Dataset.from_tensor_slices(get_filenames(is_training, data_dir))
    dataset = dataset.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(parse_record)
    if is_training:
        # prepossess
        dataset = dataset.shuffle(buffer_size=ModelConfig.num_image['train'])
        dataset = dataset.map(
          lambda image, cl_class, label: preprocess_image(image, cl_class, label, is_training))
        dataset = dataset.prefetch(batch_size)
    # repeat and batch for training
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    # return dict for model input
    labels = dict()
    images, cl_class, seg_labels = iterator.get_next()
    labels['cl_class'] = cl_class
    labels['seg_labels'] = seg_labels

    return images, labels


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # import numpy as np
    # sess = tf.InteractiveSession()
    # input_fn(False, './', 4)
    pass


