from tensorflow.python.client import device_lib
from unet_utils import *
from unet_utils import weights_dir, log_dir

from model import *

import pickle

from functools import partial

import pandas as pd
import numpy as np

import tensorflow

from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta, SGD
# K.set_image_dim_ordering('th')
# K.image_data_format('tf')
# K.tensorflow_backend.set_image_dim_ordering('tf')
K.set_image_data_format('channels_first')

# try:
#     from keras.engine import merge
# except ImportError:
#     from keras.layers.merge import concatenate


print(device_lib.list_local_devices())
print(len(device_lib.list_local_devices()))


# num_outputs, optimizer = RMSprop(lr=5e-4)):
def train_unet(model, num_outputs, load_weights_filepath=None):
    """
    takes a model and fits using callbacks
    """

    # this has the test/train ID matches
    train_val_test_dict = pickle.load(open("train_val_test_dict.pkl", "rb"))

    model_name = f"3dunet_{num_outputs}_outputs"

    if load_weights_filepath:
        # by_name=True allows you to use a different architecture and bring in the weights from the matching layers
        model.load_weights(load_weights_filepath, by_name=True)

    # Callbacks:
    early_stopping_cb = tensorflow.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=2, mode='auto')
    # cb_2 = keras.callbacks.ModelCheckpoint(filepath="./weights/3pred_weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    filename = f"model_weights_{num_outputs}_outputs.h5"
    model_checkpoint_cb = tensorflow.keras.callbacks.ModelCheckpoint(filepath=str(
        weights_dir / filename), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
    tensorboard_cb = tensorflow.keras.callbacks.TensorBoard(log_dir=str(log_dir / f"{model_name}"), histogram_freq=0, profile_batch=32, write_graph=True,
                                                            write_images=False, embeddings_freq=0, embeddings_metadata=None, update_freq='epoch')

    # Params for generators:
    params = {'dim': (160, 192, 160),
              'batch_size': 1,
              'n_classes': 3,
              'n_channels': 4,
              'shuffle': True,
              'num_outputs': num_outputs}
    # Generators:
    if num_outputs == 3:
        survival_data_df = pd.read_csv('survival_data.csv')
        sub_train_val_test_dict = {}
        sub_train_val_test_dict['train'] = [
            x for x in train_val_test_dict['val'] if x in set(survival_data_df.Brats17ID)]
        sub_train_val_test_dict['val'] = [
            x for x in train_val_test_dict['val'] if x in set(survival_data_df.Brats17ID)]
        training_generator = DataGenerator(
            sub_train_val_test_dict['train'], **params)
        validation_generator = DataGenerator(
            sub_train_val_test_dict['val'], **params)
    else:
        training_generator = DataGenerator(
            train_val_test_dict['train'], **params)
        validation_generator = DataGenerator(
            train_val_test_dict['val'], **params)

    # Fit:
    # results = model.fit_generator(generator=training_generator,
    #                               validation_data=validation_generator,
    #                               epochs=100,
    #                               nb_worker=4,
    #                               callbacks=[early_stopping_cb, model_checkpoint_cb, tensorboard_cb])


if __name__ == '__main__':

    num_outputs = 2

    model = create_model_2(input_shape=(4, 160, 192, 160),
                           n_base_filters=12,
                           depth=5,
                           dropout_rate=0.3,
                           n_segmentation_levels=3,
                           n_labels=3,
                           num_outputs=num_outputs,
                           optimizer='adam',
                           learning_rate=1e-2,
                           activation_name="sigmoid",
                           branch_enable=1)

    # train_unet(model, num_outputs)
