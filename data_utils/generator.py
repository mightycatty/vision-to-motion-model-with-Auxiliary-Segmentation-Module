import numpy as np
import os
from data_utils import dataset_util
import PIL
import tensorflow as tf
import io
# from config_old import ModelConfig
import cv2
from config_folder.config_agent import DataConfig


def resize(img, shape=(256, 256)):
    img = cv2.resize(img, shape)
    return img


def batch_resize(img_batch, resize_shape=(256, 256)):
    img_list = []
    for img_item in img_batch:
        img_list.append(resize(img_item, resize_shape))
    img_batch = np.stack(img_list, axis=0)
    img_batch = np.expand_dims(img_batch, axis=3)
    return img_batch


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
            image = resize(np.array(image))
            batch_data.append(image)
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
        if '\\' in image_dir:
            sample_name = image_dir.split('\\')[-1].split('_left')[0] + '_gtFine_labelIds.png'
        else:
            sample_name = image_dir.split('/')[-1].split('_left')[0] + '_gtFine_labelIds.png'
        sub_folder = sample_name.split('_')[0]
        prefix = os.path.join(image_dir.split('data')[0]+'data/gtFine_trainvaltest/gtFine/')
        folder = 'train' if 'train' in image_dir else 'val'
        label_dir = os.path.join(prefix, folder, sub_folder, sample_name)
        if os.path.exists(label_dir):
            return label_dir
        else:
            print('%s not found corresponding label')
            print (label_dir)
            exit(0)
            return None
    samples = dataset_util.get_file_list(data_folder)
    labels = [_get_label_for_cityscape(item) for item in samples]
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


def finetune_generator(folder='train', batch_size=16):
    """
    finetune model with indoor data
    :param folder:
    :param batch_size:
    :return:
    """
    # 255 mean invalid labels
    assert folder in ['train', 'val'], 'folder must be train or val'
    batch_size = int(batch_size // 2)
    clf_folder = os.path.join(DataConfig.clf_data_folder, folder)
    cityscape_seg_folder = os.path.join(DataConfig.cityscape_folder, folder)
    clf_gen = classification_generator(clf_folder, batch_size)
    indoor_seg_gen = indoor_seg_generator(DataConfig.seg_data_folder, DataConfig.seg_label_folder, batch_size)
    while True:
        cl_imgs, cl_labels = next(clf_gen)
        # seg_imgs, seg_labels = next(segmentation_generator(folder, batch_size))
        seg_imgs, seg_labels = next(indoor_seg_gen)
        seg_labels = np.squeeze(seg_labels)
        data = np.concatenate([cl_imgs, seg_imgs], axis=0).astype(np.float32)
        cl_labels = np.concatenate([cl_labels, 255*np.ones(shape=(batch_size, ))]).astype(np.int16)
        cl_labels = cl_labels.reshape((-1, ))
        seg_labels_scale = np.concatenate([255*np.ones(shape=(batch_size, 256, 256)), seg_labels], axis=0).astype(np.int16)
        seg_labels_scale = np.expand_dims(seg_labels_scale, axis=3)
        seg_labels_scale_0 = batch_resize(seg_labels_scale, (32, 32))
        seg_labels_scale_0[seg_labels_scale_0 > 200] = 255
        seg_labels_scale_1 = batch_resize(seg_labels_scale, (16, 16))
        seg_labels_scale_1[seg_labels_scale_1 > 200] = 255
        yield data, [cl_labels, seg_labels_scale_0, seg_labels_scale_1]


def pretrain_generator(folder='train', batch_size=16):
    """
    pretrain model with cityscape segmentation data
    :param folder:
    :param batch_size:
    :return:
    """
    # 255 mean invalid labels
    assert folder in ['train', 'val'], 'folder must be train or val'
    cityscape_seg_folder = os.path.join(DataConfig.cityscape_folder, folder)
    cityscape_seg_gen = cityscape_seg_generator(cityscape_seg_folder)
    while True:
        seg_imgs, seg_labels = next(cityscape_seg_gen)
        cl_labels = 255*np.ones(shape=(batch_size, )).astype(np.int16).reshape((-1, ))
        seg_labels_scale_0 = batch_resize(seg_labels, (32, 32))
        seg_labels_scale_0[seg_labels_scale_0 > 200] = 255
        seg_labels_scale_1 = batch_resize(seg_labels, (16, 16))
        seg_labels_scale_1[seg_labels_scale_1 > 200] = 255
        yield seg_imgs, [cl_labels, seg_labels_scale_0, seg_labels_scale_1]


if __name__ == '__main__':
    from data_utils.visualization import apply_mask
    import matplotlib.pyplot as plt
    gen = pretrain_generator(batch_size=16)
    while True:
        data, labels = next(gen)
        seg_labels = labels[-1]
        clf_labels = labels[0]
        for data_item, seg_item, clf_item in zip(data, seg_labels, clf_labels):
            color = [0, 75, 75]
            img = apply_mask(data_item, np.squeeze(seg_item), color)
            img = np.uint8(img)
            if clf_item < 255:
                clf_item = DataConfig.classification_categories[clf_item]
                plt.title(str(clf_item))
            # plt.imsave('test.png', img)
            plt.imshow(img)
            plt.show()