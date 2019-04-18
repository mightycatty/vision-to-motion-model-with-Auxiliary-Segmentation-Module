"""
utils for inference or batch evaluation
"""
import keras
import tensorflow as tf
import numpy as np

from data_utils import dataset_util
import io
import PIL
import matplotlib.pyplot as plt
from keras_implementation.model import model_fn
from tqdm import tqdm
from config_old import ModelConfig as M_old
import cv2
from keras_implementation.generator import resize
from keras_implementation.visualization import apply_mask
import shutil
import os
from keras_implementation import config
from postprocessing.mask_refine import mask_refine_entry


class ImageEvaluation(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.evaluation_list = dataset_util.get_file_list('D:\herschel\\navigation\data\classification\\val\\adjust left')

    def on_batch_end(self, batch, logs={}):
        for item in self.evaluation_list:
            with tf.gfile.GFile(item, 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = PIL.Image.open(encoded_jpg_io)
            img = np.array(image)
            img = np.expand_dims(img, axis=0)
            _, label = self.model.predict(img)
            label = np.squeeze(label)
            plt.imsave(item.plit('\\')[-1], label)


class Model(object):
    """
    load unit_model
    model_dir is list of [unit weight, segmentation weight, classification weight]
    """
    def __init__(self, model_dir):
        self.unit_model, _, _ = model_fn()
        print ('loading model...')
        for item in model_dir:
            if os.path.exists(item):
                print ('loading model from {}'.format(item))
                self.unit_model.load_weights(item, by_name=True, skip_mismatch=True)

    def inference(self, img_data):
        if len(img_data.shape) == 3:
            img_data = np.expand_dims(img_data, axis=0)
        # split batch to avoid OOM
        img_data = np.split(img_data, np.arange(0, img_data.shape[0], 20), axis=0)
        classification_result = []
        segmentation_result = []
        for item in img_data:
            if item.shape[0] > 0:
                c, s = self.unit_model.predict(item)
                classification_result.append(c)
                segmentation_result.append(s)
        classification_result = np.vstack(classification_result)
        segmentation_result = np.vstack(segmentation_result)
        return classification_result, segmentation_result


def batch_evaluation(folder):
    save_folder = 'D:\herschel\\navigation\keras_implementation\\test'
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.mkdir(save_folder)
    sample_list = dataset_util.get_file_list(folder)
    # model = Model('D:\herschel\\navigation\keras_implementation\merge_model')
    model_type = config.ModelConfig.backbone
    # model_type = 'ResNet'
    model_dir = [model_type + item for item in ['_merge_model', '_segmentation_model_on_cityscape', '_classification_model']]
    model = Model(model_dir)
    bar = tqdm(total=len(sample_list))
    color = [0, 0, 75]
    # color = [60, 40, 122]
    for item in sample_list:
        with tf.gfile.GFile(item, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        img = np.array(image)
        img = resize(img)
        # img = rotate(img, -90, reshape=True)
        classification, label = model.inference(img)
        # label = model.inference(img)
        label = np.squeeze(label)
        classification = np.argmax(classification)
        label = np.argmax(label, axis=-1)
        # label = mask_refine_entry(label.astype(np.uint8))
        img = apply_mask(img, label, color)
        img = np.uint8(img)
        # img = np.hstack([img, label_3]).astype(np.uint8)
        save_name = M_old.classification_categories[classification] + '_' + item.split('\\')[-1]
        # save_name = item.split('\\')[-1]
        save_name = os.path.join(save_folder, save_name)
        plt.imsave(save_name, img)
        bar.update()


if __name__ == '__main__':
    from scipy.ndimage.interpolation import rotate
    folder = 'D:\herschel\\navigation\data\classification\\val'
    # folder = 'D:\herschel\\navigation\data\indoor_nav\\test'
    # folder = 'D:\herschel\\navigation\data\leftImg8bit_trainvaltest\leftImg8bit\\test\\berlin'
    # folder = 'D:\herschel\\navigation\data\indoor_nav\\new\TrainingLabelData\TrainingLabelData'
    # folder = 'D:\herschel\\robot_arm\src\data\segmentation'
    batch_evaluation(folder)


