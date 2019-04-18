"""
evaluation performance for pre-trained model
Metrics: mAP, IoU, Confusion Matrix
"""
from keras_implementation.inference import Model
from sklearn.metrics import classification_report, confusion_matrix
from keras_implementation.config import ModelConfig as Modelconfig_new
from data_utils import dataset_util
import numpy as np
import PIL
import tensorflow as tf
from keras_implementation.generator import resize
import io
from config_old import ModelConfig
import matplotlib.pyplot as plt
import itertools
import keras
import seaborn as sns
from keras_implementation.train_utils import iou_metric_batch


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return


def load_data():
    """
    load data for evaluation from val folder of data set
    :return: tuple of (data, label) in numpy
    """
    # load classification data
    classification_list = dataset_util.get_file_list('D:\herschel\\navigation\data\classification\\val')
    classification_data = []
    classification_label = []
    for item in classification_list:
        # prepare label
        label_str = item.split('\\')[-2]
        if label_str in ModelConfig.classification_categories:
            classification_label.append(ModelConfig.classification_categories.index(label_str))
            # load data
            with tf.gfile.GFile(item, 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = PIL.Image.open(encoded_jpg_io)
            img = np.array(image)
            img = resize(img)
            classification_data.append(img)
    classification_data = np.stack(classification_data, axis=0)
    classification_label = np.array(classification_label)
    # load segmentation data
    segmentation_list = dataset_util.get_file_list('D:\herschel\\navigation\data\indoor_nav\\new\TrainingLabelData\TrainingLabelData')
    segmentation_data = []
    segmentation_label = []
    for item in segmentation_list:
        # load data
        with tf.gfile.GFile(item, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        img = np.array(image)
        img = resize(img)
        segmentation_data.append(img)
        # load label
        item = item.replace('Training', 'Pixel')
        item = item.replace('jpg', 'png')
        with tf.gfile.GFile(item, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        img = np.array(image)
        img[img > 0] = 1
        img = resize(img)
        segmentation_label.append(img)
    segmentation_data = np.stack(segmentation_data, axis=0)
    segmentation_label = np.stack(segmentation_label, axis=0)
    return classification_data, classification_label, segmentation_data, segmentation_label


def visualization_before_activation():
    """
    visualize the output of final feature layer before activation function of each branch
    :return:
    """
    # load data
    cl_data, cl_label, seg_data, seg_label = load_data()
    # load model for evaluation
    # model_type = Modelconfig_new.backbone
    # model_type = 'ResNet'
    model_type = 'VGG'
    # load pretrain model
    model_dir_list = [model_type + item for item in ['_segmentation_model_on_cityscape']]
    model_wrapper_pre = Model(model_dir_list)
    # load finetune model
    model_dir_list = [model_type + item for item in ['_segmentation_model', '_classification_model']]
    model_wrapper_fine = Model(model_dir_list)

    # model = model_wrapper.unit_model
    # segmentation_layer = model.get_layer('segmentation')
    # segmentation_layer.activation = None
    # classification_layer = model.get_layer('classification')
    # classification_layer.activation = None
    # model = keras.models.Model(model.input, [classification_layer.output, segmentation_layer.output])
    # model_wrapper.unit_model = model

    # result for pretrain model
    cl_result_pre, _ = model_wrapper_pre.inference(cl_data)
    cl_result_pre = cl_result_pre[:, :-1]
    # _, seg_result_pre = model_wrapper_pre.inference(seg_data)
    # seg_result_pre = seg_result_pre[:, :, :, 1]
    # result for finetune model
    cl_result_fine, _ = model_wrapper_fine.inference(cl_data)
    cl_result_fine = cl_result_fine[:, :-1]
    # _, seg_result_fine = model_wrapper_fine.inference(seg_data)
    # seg_result_fine = seg_result_fine[:, :, :, 1]
    # visualization
    # plt.subplot(121)
    yticklabels = False
    xticklabels = ['turn left', 'turn right', 'adjust left', 'adjust right', 'move forward', 'turn around']
    ax = sns.heatmap(cl_result_pre, xticklabels=xticklabels, yticklabels=yticklabels, cbar=False)
    # ax.set_title('Activation From Pretrained Model', fontsize=8)
    ax.set_ylabel('Sample Index', fontsize=15)
    # ax.set_xlabel('Softmax Activation Vector')
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=15, horizontalalignment='right', fontsize=12)
    plt.savefig(model_type + '_activation_pre.eps', format='eps', dpi=1000)
    # plt.show()
    # plt.subplot(122)
    yticklabels = False
    xticklabels = ['turn left', 'turn right', 'adjust left', 'adjust right', 'move forward', 'turn around']
    ax = sns.heatmap(cl_result_fine, xticklabels=xticklabels, yticklabels=yticklabels)
    ax.set_ylabel('Sample Index', fontsize=15)
    # ax.set_title('Activation From Finetuned Model', fontsize=8)
    # ax.set_xlabel('Softmax Activation Vector', fontsize=8)
    # 设置坐标字体方向
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=15, horizontalalignment='right', fontsize=12)
    # plt.suptitle('Variant 1: '+model_type, fontsize=16)
    plt.savefig(model_type + '_activation_fine.eps', format='eps', dpi=1000)
    return


def main():
    # load model for evaluation
    model_type = Modelconfig_new.backbone
    # model_type = 'ResNet'
    model_dir_list = [model_type + item for item in ['_merge_model', '_segmentation_model', '_classification_model']]
    model = Model(model_dir_list)
    # load data
    cl_data, cl_label, seg_data, seg_label = load_data()
    # predictive label for cl data
    cl_cl, cl_seg = model.inference(cl_data)
    cl_cl = np.argmax(cl_cl, axis=-1)
    # predictive label for seg data
    seg_cl, seg_seg = model.inference(seg_data)
    # iou = iou_metric_batch(seg_label, seg_seg)
    seg_seg = np.argmax(seg_seg, axis=-1)
    cl_cm = confusion_matrix(cl_label, cl_cl)
    # seg_cm = confusion_matrix(seg_label.flatten(), seg_seg.flatten())
    print (cl_cm)
    # print (seg_cm)
    # print (iou)


if __name__ == '__main__':
    # main()
    visualization_before_activation()


