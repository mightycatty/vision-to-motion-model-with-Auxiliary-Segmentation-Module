"""
model building with keras api
"""
import keras
import tensorflow as tf
from keras import layers, models


def defined_loss(y_true, y_pred):
    """
    defined loss for incomplete dataset, label of 255 are ignored when calculate loss
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true = keras.backend.reshape(y_true, (-1,))
    y_pred = keras.backend.reshape(y_pred, (-1, y_pred.shape[-1].value))
    valid_weight = tf.where(tf.equal(y_true, 255), keras.backend.zeros_like(y_true), keras.backend.ones_like(y_true))
    y_true = tf.where(tf.equal(y_true, 255), keras.backend.zeros_like(y_true), y_true)
    loss = keras.backend.sum(valid_weight * keras.backend.sparse_categorical_crossentropy(y_true, y_pred))
    loss = loss / (keras.backend.sum(valid_weight) + 1e-6)
    return loss


def navigation_model(input_tensor=None):
    """
    build navigation model
    :param input_tensor:
    :param weights_path:
    :return:
    """
    if input_tensor is None:
        input_tensor = layers.Input(shape=(256, 256, 3))
    # block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(input_tensor)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x_block4 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    # segmentation module 0
    mask_0 = layers.Conv2D(2, (3, 3),
                           activation='sigmoid',
                           padding='same',
                           name='mask_0')(x_block4)
    feature_fusion_0 = layers.Concatenate()([x_block4, mask_0])
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(feature_fusion_0)
    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x_block5 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    # segmentation module 1
    mask_1 = layers.Conv2D(2, (3, 3),
                           activation='sigmoid',
                           padding='same',
                           name='mask_1')(x_block5)
    feature_fusion_1 = layers.Concatenate()([x_block5, mask_1])
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(feature_fusion_1)
    # Classification block
    classes = 6
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(1024, activation='relu', name='fc1')(x)
    x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    # Create model.
    model = models.Model(input_tensor, [x, mask_0, mask_1], name='navigation_model')
    return model


def compile_model(base_model):
    """
    compile model
    defining LOSS
    :return:
    """
    clf_out, mask_0, mask_1 = base_model.outputs
    # summary
    tf.summary.image('clf_out', clf_out, max_outputs=6)  # Concatenate row-wise.
    tf.summary.image('mask_0', tf.concat([255 * tf.cast(mask_0, tf.float32)] * 3, axis=3),
                     max_outputs=6)  # Concatenate row-wise.
    tf.summary.image('mask_1', tf.concat([255 * tf.cast(mask_1, tf.float32)] * 3, axis=3),
                     max_outputs=6)  # Concatenate row-wise.
    base_model.compile(
        loss=[keras.losses.sparse_categorical_crossentropy, defined_loss, defined_loss],
        optimizer='adam', #keras.optimizers.SGD(lr=3e-5, momentum=0.9),
        metrics=['acc']
    )
    return base_model


if __name__ == '__main__':
    # plot model
    u = navigation_model()
    from keras.utils import plot_model
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    plot_model(u, show_shapes=True, to_file='united_model.png')


