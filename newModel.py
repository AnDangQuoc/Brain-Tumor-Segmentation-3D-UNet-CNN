from keras.engine import Model
import keras
from keras.engine import Input, Model
from keras.optimizers import Adam, RMSprop, Adadelta, SGD
from keras.initializers import he_normal
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D, Flatten, Dense, GlobalAveragePooling3D, concatenate, Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D
from keras import backend as K

from unet_utils import *


def create_new_model(input_shape=(4, 160, 192, 160),
                     n_base_filters=12,
                     depth=5,
                     dropout_rate=0.3,
                     n_segmentation_levels=3,
                     n_labels=3,
                     num_outputs=3,
                     optimizer='adam',
                     learning_rate=1e-3,
                     activation_name="sigmoid",
                     n_branch=1):
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
    :param n_branch: num of upscale branch
    :return:
    """

    if optimizer.lower() == 'adam':
        optimizer = Adam(lr=learning_rate)

    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    # level 0->4 (down sampling)
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(
                current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(
            in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    last_layer = current_layer
    segmentation_layers = list()
    segmentation_layers_upsampling = list()
    # 3->0
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(
            current_layer, level_filters[level_number])
        segmentation_layers_upsampling.insert(0, up_sampling)
        concatenation_layer = concatenate(
            [level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module(
            concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, create_convolution_block(
                current_layer, n_filters=n_labels, kernel=(1, 1, 1)))

    segmentation_layers_2 = list()
    segmentation_layers_upsampling_2 = list()
    current_layer = last_layer
    if n_branch >= 2:
        for level_number in range(depth - 2, -1, -1):
            up_sampling = create_up_sampling_module(
                current_layer, level_filters[level_number])
            segmentation_layers_upsampling_2.insert(0, up_sampling)
            concatenation_layer = concatenate(
                [segmentation_layers_upsampling[level_number], level_output_layers[level_number], up_sampling], axis=1)
            localization_output = create_localization_module(
                concatenation_layer, level_filters[level_number])
            current_layer = localization_output
            if level_number < n_segmentation_levels:
                segmentation_layers_2.insert(0, create_convolution_block(
                    current_layer, n_filters=n_labels, kernel=(1, 1, 1)))

    current_layer = last_layer
    segmentation_layers_3 = list()
    segmentation_layers_upsampling_3 = list()
    if n_branch == 3:
        for level_number in range(depth - 2, -1, -1):
            up_sampling = create_up_sampling_module(
                current_layer, level_filters[level_number])
            segmentation_layers_upsampling_3.insert(0, up_sampling)
            concatenation_layer = concatenate(
                [segmentation_layers_upsampling_2[level_number], segmentation_layers_upsampling[level_number], level_output_layers[level_number], up_sampling], axis=1)
            localization_output = create_localization_module(
                concatenation_layer, level_filters[level_number])
            current_layer = localization_output
            if level_number < n_segmentation_levels:
                segmentation_layers_3.insert(0, create_convolution_block(
                    current_layer, n_filters=n_labels, kernel=(1, 1, 1)))

    output_layer = None
    # 3->0
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]

        if n_branch == 3:
            segmentation_layer_2 = segmentation_layers_2[level_number]
            segmentation_layer_3 = segmentation_layers_3[level_number]
            final_segment_layer = Add()(
                [segmentation_layer, segmentation_layer_2, segmentation_layer_3])
        elif n_branch == 2:
            segmentation_layer_2 = segmentation_layers_2[level_number]
            final_segment_layer = Add()(
                [segmentation_layer, segmentation_layer_2])
        else:
            final_segment_layer = segmentation_layer

        if output_layer is None:
            output_layer = final_segment_layer
        else:
            output_layer = Add()([output_layer, final_segment_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(
        activation=activation_name, name='activation_block')(output_layer)

    model = Model(inputs=inputs, outputs=[activation_block])

    model.compile(optimizer=RMSprop(lr=5e-4),
                  loss={'activation_block': weighted_dice_coefficient_loss},
                  loss_weights={'activation_block': 1.},
                  metrics={'activation_block': ['accuracy', weighted_dice_coefficient, dice_coefficient]})

    # keras.utils.plot_model(model, "my_first_model.png")

    # add the parameter that allows me to show everything instead of cutting it off
    model.summary(line_length=150)
    return model
    # model.save_weights("./weights/model_3_weights.h5")
    # print("Saved model to disk")


def create_new_model_segment_only(input_shape=(4, 160, 192, 160),
                                  n_base_filters=12,
                                  depth=5,
                                  dropout_rate=0.3,
                                  n_segmentation_levels=3,
                                  n_labels=3,
                                  num_outputs=3,
                                  optimizer='adam',
                                  learning_rate=1e-3,
                                  activation_name="sigmoid",
                                  n_branch=1):
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
    :param n_branch: num of upscale branch
    :return:
    """

    if optimizer.lower() == 'adam':
        optimizer = Adam(lr=learning_rate)

    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    # level 0->4 (down sampling)
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(
                current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(
            in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    segmentation_layers_upsampling = list()
    segmentation_layers_2 = list()
    segmentation_layers_3 = list()
    segmentation_layers_upsampling_3 = list()

    # 3->0
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(
            current_layer, level_filters[level_number])
        segmentation_layers_upsampling.insert(0, up_sampling)
        concatenation_layer = concatenate(
            [level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module(
            concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, create_convolution_block(
                current_layer, n_filters=n_labels, kernel=(1, 1, 1)))
            if n_branch >= 2:
                segmentation_layers_2.insert(0, create_convolution_block(
                    current_layer, n_filters=n_labels, kernel=(1, 1, 1)))
            if n_branch == 3:
                segmentation_layers_3.insert(0, create_convolution_block(
                    current_layer, n_filters=n_labels, kernel=(1, 1, 1)))

    output_layer = None
    # 3->0
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]

        if n_branch == 3:
            segmentation_layer_2 = segmentation_layers_2[level_number]
            segmentation_layer_3 = segmentation_layers_3[level_number]
            final_segment_layer = Add()(
                [segmentation_layer, segmentation_layer_2, segmentation_layer_3])
        elif n_branch == 2:
            segmentation_layer_2 = segmentation_layers_2[level_number]
            final_segment_layer = Add()(
                [segmentation_layer, segmentation_layer_2])
        else:
            final_segment_layer = segmentation_layer

        if output_layer is None:
            output_layer = final_segment_layer
        else:
            output_layer = Add()([output_layer, final_segment_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(
        activation=activation_name, name='activation_block')(output_layer)

    model = Model(inputs=inputs, outputs=[activation_block])

    model.compile(optimizer=RMSprop(lr=5e-4),
                  loss={'activation_block': weighted_dice_coefficient_loss},
                  loss_weights={'activation_block': 1.},
                  metrics={'activation_block': ['accuracy', weighted_dice_coefficient, dice_coefficient]})

    keras.utils.plot_model(model, "my_second_model.png")

    # add the parameter that allows me to show everything instead of cutting it off
    model.summary(line_length=150)

    return model
    # model.save_weights("./weights/model_3_weights.h5")
    # print("Saved model to disk")


def create_flatten_model(input_shape=(4, 160, 192, 160),
                         n_base_filters=12,
                         depth=5,
                         dropout_rate=0.3,
                         n_segmentation_levels=3,
                         n_labels=3,
                         num_outputs=3,
                         optimizer='adam',
                         learning_rate=1e-3,
                         activation_name="sigmoid",
                         n_branch=1):

    return
    # K.set_image_dim_ordering('th')
    # K.image_data_format('tf')
    # K.tensorflow_backend.set_image_dim_ordering('tf')
    # K.set_image_data_format('channels_first')


# create_flatten_model(input_shape=(4, 160, 192, 160),
#                      n_base_filters=12,
#                      depth=5,
#                      dropout_rate=0.3,
#                      n_segmentation_levels=3,
#                      n_labels=3,
#                      num_outputs=1,
#                      optimizer='adam',
#                      learning_rate=1e-2,
#                      activation_name="sigmoid",
#                      n_branch=3)
