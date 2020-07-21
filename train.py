from tensorflow.python.client import device_lib
from unet_utils import *
from unet_utils import weights_dir, log_dir

import pickle
import inquirer


from functools import partial

import pandas as pd
import numpy as np

from keras import backend as K
from keras.engine import Model
from keras.optimizers import Adam, RMSprop, Adadelta, SGD

from newModel import *

K.tensorflow_backend.set_image_dim_ordering('tf')
K.set_image_data_format('channels_first')

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


def train_model(model, modelName, n_branch, load_weights_filepath=None):
    """
    takes a model and fits using callbacks
    """

    # this has the test/train ID matches
    train_val_test_dict = pickle.load(open("train_val_test_dict.pkl", "rb"))

    model_name = f"{modelName}_ouputs"

    if load_weights_filepath:
        # by_name=True allows you to use a different architecture and bring in the weights from the matching layers
        model.load_weights(load_weights_filepath, by_name=True)

    # Callbacks:
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=2, mode='auto')
    # cb_2 = keras.callbacks.ModelCheckpoint(filepath="./weights/3pred_weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    filename = f"model_weights_{modelName}_outputs_{n_branch}.h5"
    model_checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=str(
        weights_dir / modelName / filename), monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    tensorboard_cb = keras.callbacks.TensorBoard(log_dir=str(log_dir / f"{model_name}"), histogram_freq=0, batch_size=32, write_graph=True,
                                                 write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

    # Params for generators:
    params = {'dim': (160, 192, 160),
              'batch_size': 1,
              'n_classes': 3,
              'n_channels': 4,
              'shuffle': True,
              'num_outputs': 1,
              'n_branch': n_branch}

    training_generator = DataGenerator(
        train_val_test_dict['train'], **params)
    validation_generator = DataGenerator(
        train_val_test_dict['val'], **params)

    # Fit:
    results = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=1,
                                  workers=0,
                                  callbacks=[early_stopping_cb, model_checkpoint_cb, tensorboard_cb])


if __name__ == '__main__':
    questions = [
        inquirer.List('model',
                      message="What model do you want to train?",
                      choices=['first_model', 'second_model'],
                      ),
        inquirer.List(
            'n_branch', message="How many branch do you want to train?", choices=[1, 2, 3])
    ]
    answers = inquirer.prompt(questions)

    modelName = answers["model"]
    n_branch = int(answers["n_branch"])
    if modelName == "first_model":
        model = create_new_model(input_shape=(4, 160, 192, 160),
                                 n_base_filters=12,
                                 depth=5,
                                 dropout_rate=0.3,
                                 n_segmentation_levels=3,
                                 n_labels=3,
                                 num_outputs=1,
                                 optimizer='adam',
                                 learning_rate=1e-2,
                                 activation_name="sigmoid",
                                 n_branch=n_branch)
    else:
        model = create_new_model_segment_only(input_shape=(4, 160, 192, 160),
                                              n_base_filters=12,
                                              depth=5,
                                              dropout_rate=0.3,
                                              n_segmentation_levels=3,
                                              n_labels=3,
                                              num_outputs=1,
                                              optimizer='adam',
                                              learning_rate=1e-2,
                                              activation_name="sigmoid",
                                              n_branch=n_branch)

    load_weights_filepath = None

    if n_branch >= 2:
        load_weights_filepath = filepath = str(
            weights_dir / answers["model"] / f"model_weights_{modelName}_outputs_{n_branch-1}.h5")

    train_model(model, modelName, n_branch, load_weights_filepath)
