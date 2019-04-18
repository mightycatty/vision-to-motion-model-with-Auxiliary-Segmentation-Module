class ModelConfig(object):
    data_dir = None
    # train_tfrecord_dir = 'D:\herschel\\navigation\\tf_records\\fine_combine_train.record'
    # val_tfrecord_dir = 'D:\herschel\\navigation\\tf_records\\fine_combine_val.record'
    train_tfrecord_dir = 'D:\herschel\\navigation\\tf_records\\cl_train.record'
    val_tfrecord_dir = 'D:\herschel\\navigation\\tf_records\\cl_val.record'
    # debug = 'store_true'
    debug = False
    interest_label = [6, 7, 8, 9, 10] # road/flat/sidewalk etc.
    # interest_label = [10]
    seg_num_classes = 2 # foreground/background
    classification_categories = ['turn_left', 'turn_right', 'adjust_left', 'adjust_right', 'move_forward', 'turn_around', 'target_found']
    classification_num_classes = len(classification_categories)
    height = 512
    width = 512
    depth = 3
    min_scale = 0.5
    max_scale = 2.0
    bn_decay = 0.9997
    num_image = {
        'train': 3985,
        'validation': 1500,
    }


class TrainingConfig(object):
    clean_model_dir = 0
    train_epochs = 100
    epochs_per_eval = 1
    tensorboard_images_max_outputs = 6
    batch_size = 16
    learning_rate_policy = 'poly' # choices=['poly', 'piecewise']
    max_iter = 30000000
    base_architecture = 'resnet_v2_50' # choices=['resnet_v2_50', 'resnet_v2_101']
    lr_for_classification = 1e-3
    initial_learning_rate = 7e-5
    end_learning_rate = 1e-6
    initial_global_step = 0
    power = 0.9
    momentum = 0.9
    weight_decay = 3e-4 # regulization
    freeze_batch_norm = False
    model_dir = './merge'




