import numpy as np
import os
from data_utils import dataset_util
import PIL
import tensorflow as tf
import io
# from config_old import ModelConfig
import cv2


def resize(img, shape=(512, 512)):
    img = cv2.resize(img, shape)
    return img


def interested_labels(label):
    """
    set interested labels in segmentation as forground and the rest as backgound
    :label: numpy array
    :return:
    """
    interested_label_index = DataConfig.interest_label
    for item in interested_label_index:
        label[label == item] = 1
    label[label != 1] = 0
    return label


def classification_generator(data_folder, batch_size=16):
    """
    generator for vision-to-motion classification task
    :param data_folder:
    :param batch_size:
    :return:
    """
    class_list = DataConfig.classification_categories
    # data_folder = 'D:\herschel\\navigation\data\classification\\' + data_folder
    samples = []
    labels = []
    for item in class_list:
        path = os.path.join(data_folder, item)
        item_list = (dataset_util.get_file_list(path))
        samples += item_list
        labels += [class_list.index(item)]*len(item_list)
    samples = np.array(samples)
    labels = np.array(labels)
    shuffle_index = np.arange(0, len(samples))
    np.random.shuffle(shuffle_index)
    samples = (samples[shuffle_index])
    labels = list(labels[shuffle_index])
    while True:
        batch_index = np.random.randint(0, len(samples), batch_size)
        batch_data_list = list(samples[batch_index])
        batch_data = []
        batch_label = [labels[item] for item in batch_index]
        for path_item in batch_data_list:
            with tf.gfile.GFile(path_item, 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = PIL.Image.open(encoded_jpg_io)
            batch_data.append(np.array(image))
        batch_data = np.stack(batch_data, axis=0)
        batch_label = np.array(batch_label)
        yield batch_data, batch_label


def cityscape_seg_generator(data_folder, batch_size=16):
    """
    generate data for training from cityscape dataset
    :param data_folder:
    :param batch_size:
    :return:
    """
    def _get_label_for_cityscape(image_dir):
        """
        get corresponding label path given image
        :return:
        """
        sample_name = image_dir.split('\\')[-1].split('_left')[0] + '_gtFine_labelIds.png'
        sub_folder = sample_name.split('_')[0]
        prefix = os.path.join(os.path.split(image_dir)[0]+'data\\gtFine_trainvaltest\\gtFine\\')
        folder = 'train' if 'train' in image_dir else 'val'
        label_dir = os.path.join(prefix, folder, sub_folder, sample_name)
        if os.path.exists(label_dir):
            return label_dir
        else:
            print('%s not found corresponding label')
            return None
    # samples = dataset_util.get_file_list('D:\herschel\\navigation\data\leftImg8bit_trainvaltest\leftImg8bit\\'+data_folder)
    samples = dataset_util.get_file_list(data_folder)
    labels = [_get_label_for_cityscape(item, data_folder) for item in samples]
    samples = np.array(samples)
    labels = np.array(labels)
    shuffle_index = np.arange(0, len(samples))
    np.random.shuffle(shuffle_index)
    samples = (samples[shuffle_index])
    labels = (labels[shuffle_index])
    while True:
        batch_index = np.random.randint(0, len(samples), batch_size)
        batch_data_list = list(samples[batch_index])
        batch_label_list = list(labels[batch_index])
        batch_data = []
        batch_label = []
        for img_item, label_item in zip(batch_data_list, batch_label_list):
            # read img
            with tf.gfile.GFile(img_item, 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = np.array(PIL.Image.open(encoded_jpg_io))
            image = resize(image)
            batch_data.append(image)
            # read labels
            with tf.gfile.GFile(label_item, 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            label = np.array(PIL.Image.open(encoded_jpg_io))
            label = resize(label)
            label = interested_labels(label)
            batch_label.append(label)
        batch_data = np.stack(batch_data, axis=0).astype(np.float32)
        batch_label = np.stack(batch_label, axis=0).astype(np.float32)
        batch_label = np.expand_dims(batch_label, 3)
        yield batch_data, batch_label


def indoor_seg_generator(data_folder, label_folder, batch_size=16):
    """
    segmentation generator for indoor obstacle avoidance segmentation map
    :param batch_size:
    :return:
    """
    samples = dataset_util.get_file_list(data_folder)
    labels = dataset_util.get_file_list(label_folder)
    samples = np.array(samples)
    labels = np.array(labels)
    shuffle_index = np.arange(0, len(samples))
    np.random.shuffle(shuffle_index)
    samples = (samples[shuffle_index])
    labels = (labels[shuffle_index])
    while True:
        batch_index = np.random.randint(0, len(samples), batch_size)
        batch_data_list = list(samples[batch_index])
        batch_label_list = list(labels[batch_index])
        batch_data = []
        batch_label = []
        for img_item, label_item in zip(batch_data_list, batch_label_list):
            # read img
            with tf.gfile.GFile(img_item, 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = np.array(PIL.Image.open(encoded_jpg_io))
            image = resize(image)
            batch_data.append(image)
            # read labels
            with tf.gfile.GFile(label_item, 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            label = np.array(PIL.Image.open(encoded_jpg_io))
            label[label > 0] = 1
            label = np.squeeze(label)
            label = resize(label)
            label = interested_labels(label)
            batch_label.append(label)
        batch_data = np.stack(batch_data, axis=0).astype(np.float32)
        batch_label = np.stack(batch_label, axis=0).astype(np.float32)
        batch_label = np.expand_dims(batch_label, 3)
        yield batch_data, batch_label


def combine_generator(folder='train', batch_size=16):
    # 255 mean invalid labels
    assert folder in ['train', 'val'], 'folder must be train or val'
    batch_size = int(batch_size // 2)
    i = 0
    while True:
        cl_imgs, cl_labels = next(classification_generator(folder, batch_size))
        seg_imgs, seg_labels = next(segmentation_generator(folder, batch_size))
        seg_imgs, seg_labels = next(segmentation_generator_test(batch_size))
        seg_labels = np.squeeze(seg_labels)
        data = np.concatenate([cl_imgs, seg_imgs], axis=0).astype(np.float32)
        cl_labels = np.concatenate([cl_labels, 255*np.ones(shape=(batch_size, ))]).astype(np.int16)
        cl_labels = cl_labels.reshape((-1, ))
        seg_labels = np.concatenate([255*np.ones(shape=(batch_size, 512, 512)), seg_labels], axis=0).astype(np.int16)
        seg_labels = np.expand_dims(seg_labels, axis=3)
        yield data, [cl_labels, seg_labels]


if __name__ == '__main__':
    from data_utils.visualization import apply_mask
    import matplotlib.pyplot as plt
    from config_folder.config import DataConfig
    while True:
        data, labels = next(indoor_seg_generator(DataConfig.seg_data_folder, DataConfig.seg_label_folder), 4)
        if np.sum(labels) > 0:
            color = [0, 75, 75]
            img = apply_mask(data[0], np.squeeze(labels[0]), color)
            img = np.uint8(img)
            # plt.imsave('test.png', img)
            plt.imshow(img)
            plt.show()
        else:
            continue