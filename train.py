"""
training model
"""
from data_utils import generator
from model import navigation_model, compile_model
from train_utils import create_callbacks
import os


def main():
    # build and compile model
    unit_model = navigation_model()
    unit_model = compile_model(unit_model)
    # load weights
    # if os.path.exists(unit_model_dir):
    #     print ('loading model from {}'.format(unit_model_dir))
    #     unit_model.load_weights(unit_model_dir, by_name=True, skip_mismatch=True)
    # if training_type == 'unit':
    # create callbacks
    model_name = 'navigation_model'
    callbacks = create_callbacks(model_name)
    # data generator
    train_generator = generator.combine_generator(folder='train', batch_size=16)
    val_generator = generator.combine_generator(folder='val', batch_size=16)
    # training
    print ('start training...')
    unit_model.fit_generator(generator=train_generator,
                         steps_per_epoch=300,
                         epochs=30,
                         verbose=1,
                        callbacks=callbacks,
                        validation_data=val_generator,
                        validation_steps=1,
                         )
    unit_model.save(model_name)


if __name__ == '__main__':
    main()