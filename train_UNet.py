from unet_utils import *
from unet_utils import weights_dir, log_dir

import pickle

from functools import partial

import pandas as pd
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.engine import Model
from keras.optimizers import Adam, RMSprop, Adadelta, SGD
import keras.backend.tensorflow_backend as tfback

# K.set_image_dim_ordering('th')
# K.image_data_format('tf')
# K.tensorflow_backend.set_image_dim_ordering('tf')
K.set_image_data_format('channels_first')

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate

from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# print(len(device_lib.list_local_devices()))

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

# Fix pacakge version error
tfback._get_available_gpus = _get_available_gpus

def train_unet(model, num_outputs, load_weights_filepath=None): # num_outputs, optimizer = RMSprop(lr=5e-4)):
    """
    takes a model and fits using callbacks
    """

    train_val_test_dict = pickle.load(open( "train_val_test_dict.pkl", "rb" ) ) # this has the test/train ID matches

    model_name = f"3dunet_{num_outputs}_outputs"

    if load_weights_filepath:
        model.load_weights(load_weights_filepath, by_name=True) # by_name=True allows you to use a different architecture and bring in the weights from the matching layers 
    
    # Callbacks:
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=2, mode='auto')
    # cb_2 = keras.callbacks.ModelCheckpoint(filepath="./weights/3pred_weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    filename = f"model_weights_{num_outputs}_outputs.h5"
    model_checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath= str(weights_dir / filename), monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    tensorboard_cb = keras.callbacks.TensorBoard(log_dir= str(log_dir / f"{model_name}"), histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    # Params for generators:
    params = {'dim': (160,192,160),
                            'batch_size': 1,
                            'n_classes': 3,
                            'n_channels': 4,
                            'shuffle': True,
                            'num_outputs': num_outputs}
    # Generators:
    if num_outputs==3:
        survival_data_df = pd.read_csv('survival_data.csv')
        sub_train_val_test_dict = {}
        sub_train_val_test_dict['train'] = [x for x in train_val_test_dict['val'] if x in set(survival_data_df.Brats17ID)]
        sub_train_val_test_dict['val'] = [x for x in train_val_test_dict['val'] if x in set(survival_data_df.Brats17ID)]
        training_generator = DataGenerator(sub_train_val_test_dict['train'], **params)
        validation_generator = DataGenerator(sub_train_val_test_dict['val'], **params)
    else:
        training_generator = DataGenerator(train_val_test_dict['train'], **params)
        validation_generator = DataGenerator(train_val_test_dict['val'], **params)

    # Fit:
    results = model.fit(x=training_generator,
                        validation_data=validation_generator,
                    epochs=10, 
                    workers=4,
                    callbacks=[early_stopping_cb, model_checkpoint_cb, tensorboard_cb])

if __name__ == '__main__':
    
    num_outputs = 1

    model = create_model(input_shape=(4, 160, 192, 160),
        n_base_filters=12,
        depth=5,
        dropout_rate=0.3,
        n_segmentation_levels=3,
        n_labels=3,
        num_outputs=num_outputs,
        optimizer='adam',
        learning_rate=1e-2,
        activation_name="sigmoid")
    
    train_unet(model, num_outputs)
