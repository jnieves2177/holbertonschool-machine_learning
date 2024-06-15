#!/usr/bin/env python3
"""This script creates an inception block"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Function to create  a transition layer
    Args:
        X: the output from the previous layer
        nb_filters: integer representing the number
                    of filters in X
        compression: compression factor for the transition layer

    Returns: The output of the transition layer and the number
             of filters within the output, respectively

    """
    init = K.initializers.he_normal()
    nfilter = int(nb_filters * compression)

    batch1 = K.layers.BatchNormalization()(X)

    relu1 = K.layers.Activation('relu')(batch1)

    conv = K.layers.Conv2D(filters=nfilter,
                           kernel_size=1, padding='same',
                           kernel_initializer=init)(relu1)
    avg_pool = K.layers.AveragePooling2D(pool_size=2, strides=2,
                                         padding='same')(conv)
    return avg_pool, nfilter
