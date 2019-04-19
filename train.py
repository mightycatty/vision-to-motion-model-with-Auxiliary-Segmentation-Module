"""
training model
"""
from data_utils import generator
from model import navigation_model, compile_model
from train_utils import create_callbacks
from config_folder.config import DataConfig, TrainingConfig
from keras.utils import multi_gpu_model


def main():
    # build and compile model
    unit_model = navigation_model()
    unit_model = compile_model(unit_model)
    unit_model = multi_gpu_model(unit_model, gpus=4)
    # load weights
    # unit_model_dir = TrainingConfig.pretrained_weights_dir
    unit_model_dir = None
    # if (unit_model_dir is not None) or (os.path.exists(unit_model_dir)):
    #     print ('loading model from {}'.format(unit_model_dir))
    #     unit_model.load_weights(unit_model_dir, by_name=True, skip_mismatch=True)
    # create callbacks
    model_name = 'navigation_model'
    callbacks = create_callbacks(model_name)
    # data generator
    train_generator = generator.pretrain_generator('train', TrainingConfig.batch_size)
    val_generator = generator.pretrain_generator(folder='val', batch_size=16)
    # training
    print ('start training...')
    unit_model.fit_generator(generator=train_generator,
                         steps_per_epoch=TrainingConfig.steps_per_epoch,
                         epochs=TrainingConfig.epochs,
                         verbose=1,
                        callbacks=callbacks,
                        validation_data=val_generator,
                        validation_steps=1,
                         )
    unit_model.save_weights(model_name)


if __name__ == '__main__':
    main()