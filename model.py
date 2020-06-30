from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D, BatchNormalization, Activation, SpatialDropout3D, Add, UpSampling3D, concatenate
from unet_utils import *
import tensorflow as tf


def create_convolution_block_2(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                               padding='same', strides=(1, 1, 1), instance_normalization=False):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding='same', kernel_initializer='he_normal',
                   strides=strides, activation="relu")(input_layer)

    return layer
    # if batch_normalization:
    #     layer = BatchNormalization(axis=1)(layer)
    # elif instance_normalization:
    #     try:
    #         from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
    #     except ImportError:
    #         raise ImportError("Install keras_contrib in order to use instance normalization."
    #                           "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
    #     layer = InstanceNormalization(axis=1)(layer)
    # if activation is None:
    #     return Activation('relu')(layer)
    # else:
    #     return activation()(layer)


def create_context_module_2(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block_2(
        input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(
        rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block_2(
        input_layer=dropout, n_filters=n_level_filters)
    return convolution2


def create_up_sampling_module_2(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block_2(up_sample, n_filters)
    return convolution


def create_localization_module_2(input_layer, n_filters):
    convolution1 = create_convolution_block_2(input_layer, n_filters)
    convolution2 = create_convolution_block_2(
        convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_model_2(input_shape=(4, 160, 192, 160),
                   n_base_filters=12,
                   depth=5,
                   dropout_rate=0.3,
                   n_segmentation_levels=3,
                   n_labels=3,
                   num_outputs=3,
                   optimizer='adam',
                   learning_rate=1e-3,
                   activation_name="sigmoid",
                   branch_enable=1):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf
    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf
    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """

    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()

    # Encoder Branch
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block_2(
                current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block_2(
                current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module_2(
            in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    # Decoder Branch
    segmentation_layers = list()

    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module_2(
            current_layer, level_filters[level_number])
        concatenation_layer = concatenate(
            [level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module_2(
            concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, create_convolution_block_2(
                current_layer, n_filters=n_labels, kernel=(1, 1, 1)))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(activation_name)(output_layer)
    # activation_block = Conv3D()

    model = Model(inputs=inputs, outputs=activation_block)

    model.compile(optimizer='adam',
                  loss=weighted_dice_coefficient_loss,
                  metrics=['accuracy'])

    print(model.summary(line_length=150))

    # tf.keras.utils.plot_model(model, show_shapes=True)
    return model
