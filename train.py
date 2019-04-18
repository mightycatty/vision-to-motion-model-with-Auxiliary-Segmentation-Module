"""
training model
"""
from data_utils import generator
from model import navigation_model, compile_model
from train_utils import create_callbacks
from config import DataConfig, ModelConfig, TrainingConfig
import os


def main():
    # build and compile model
    unit_model = navigation_model()
    unit_model = compile_model(unit_model)
    # load weights
    unit_model_dir = TrainingConfig.pretrained_weights_dir
    if (unit_model_dir is not None) or (os.path.exists(unit_model_dir)):
        print ('loading model from {}'.format(unit_model_dir))
        unit_model.load_weights(unit_model_dir, by_name=True, skip_mismatch=True)
    # create callbacks
    model_name = 'navigation_model'
    callbacks = create_callbacks(model_name)
    # data generator
    train_generator = generator.multi_task_generator(DataConfig.clf_data_folder, DataConfig.seg_data_folder,
                                                     DataConfig.seg_label_folder, TrainingConfig.batch_size)
    # val_generator = generator.multi_task_generator(folder='val', batch_size=16)
    # training
    print ('start training...')
    unit_model.fit_generator(generator=train_generator,
                         steps_per_epoch=TrainingConfig.steps_per_epoch,
                         epochs=TrainingConfig.epochs,
                         verbose=1,
                        callbacks=callbacks,
                        # validation_data=val_generator,
                        # validation_steps=1,
                         )
    unit_model.save(model_name)


if __name__ == '__main__':
    main()