"""
config for training and model
"""


class DataConfig(object):
    data_dir = None
    train_tfrecord_dir = 'D:\herschel\\navigation\\tf_records\\cl_train.record'
    val_tfrecord_dir = 'D:\herschel\\navigation\\tf_records\\cl_val.record'
    # debug = 'store_true'
    debug = False

    # config for cityscape data set
    interest_label = [6, 7, 8, 9, 10] # road/flat/sidewalk etc.

    # configuration for indoor data set
    classification_categories = ['turn_left', 'turn_right', 'adjust_left', 'adjust_right', 'move_forward', 'turn_around']
    classification_num_classes = len(classification_categories)
    clf_data_folder = 'F:\heshuai\lab\paper-for-sj\\1\code\\navigation\data\classification\\train'
    seg_label_folder = 'F:\heshuai\lab\paper-for-sj\\1\code\\navigation\data\indoor_nav\\new\PixelLabelData\PixelLabelData'
    seg_data_folder = 'F:\heshuai\lab\paper-for-sj\\1\code\\navigation\data\indoor_nav\\new\TrainingLabelData\TrainingLabelData'
    # config for input image
    image_size = (256, 256)


class ModelConfig(object):
    backbone = ['VGG', 'ResNet'][0]
    # unfreeze conv layers to P5 feature level (5mean downsample 5 time to the input image)
    backbone_trainable = False


class TrainingConfig(object):
    pretrained_weights_dir = 'F:\heshuai\lab\paper-for-sj\\1\code\\navigation\weights\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    batch_size = 4
    steps_per_epoch = 300
    epochs = 30


# prepare gpu for training
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"