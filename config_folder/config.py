"""
config for training and model
"""


class DataConfig(object):
    data_dir = None
    # debug = 'store_true'
    debug = False

    # config for cityscape data set
    interest_label = [6, 7, 8, 9, 10] # road/flat/sidewalk etc.

    # configuration for indoor data set
    import sys
    import os
    prefix = os.getcwd()
    classification_categories = ['turn_left', 'turn_right', 'adjust_left', 'adjust_right', 'move_forward', 'turn_around']
    classification_num_classes = len(classification_categories)
    clf_data_folder = 'Z:\py_project\\vision-to-motion-model-with-Auxiliary-Segmentation-Module\data\classification'
    seg_label_folder = 'Z:\py_project\\vision-to-motion-model-with-Auxiliary-Segmentation-Module\data\indoor_nav\\new\PixelLabelData\PixelLabelData'
    seg_data_folder = 'Z:\py_project\\vision-to-motion-model-with-Auxiliary-Segmentation-Module\data\indoor_nav\\new\TrainingLabelData\TrainingLabelData'
    cityscape_folder = 'Z:\py_project\\vision-to-motion-model-with-Auxiliary-Segmentation-Module\data\leftImg8bit_trainvaltest\leftImg8bit'
    # config for input image
    image_size = (256, 256)


class ModelConfig(object):
    backbone = ['VGG', 'ResNet'][0]
    # unfreeze conv layers to P5 feature level (5mean downsample 5 time to the input image)
    backbone_trainable = False


class TrainingConfig(object):
    # pretrained_weights_dir = os.path.join(prefix, 'weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    pretrained_weights_dir = None
    batch_size = 32
    steps_per_epoch = 300
    epochs = 30


if __name__ == '__main__':
    import sys
    import os
    print (os.path.split(os.path.realpath(sys.argv[0]))[0])
    print (os.getcwd())
# prepare gpu for training
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"