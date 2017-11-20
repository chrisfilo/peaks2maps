import tensorflow as tf
from functools import reduce
import math
import numpy as np


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[-1]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

def get_inference(input_images_placeholder, input_shape):
    """Build the MNIST model up to where it may be used for inference.
    Args:

    Returns:

    """
    ngf = 64
    layers = []

    input_images_placeholder = tf.expand_dims(input_images_placeholder, -1)

    conv_args = {"strides": 2,
                 "kernel_size": 4,
                 "padding": "valid",
                 "activation": tf.nn.leaky_relu,
                 "kernel_initializer": tf.random_normal_initializer(0, 0.02),
                 "name": "conv",
                 "use_bias": False}

    deconv_args = conv_args.copy()
    deconv_args["padding"] = "same"
    deconv_args["name"] = "deconv"

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        padded_input = tf.pad(input_images_placeholder,
                              [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]],
                              mode="CONSTANT")
        output = tf.layers.conv3d(padded_input, ngf, **conv_args)
        layers.append(output)

    layer_specs = [
        ngf * 2,
        # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        ngf * 2,
        # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        ngf * 4,
        #encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        ngf * 8,
        # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        # ngf * 8,
        # # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            padded_input = tf.pad(layers[-1],
                                  [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]],
                                  mode="CONSTANT")
            convolved = tf.layers.conv3d(padded_input, out_channels,
                                         **conv_args)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        # (ngf * 8, 0.5),
        # # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.5),
        # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 4, 0.5),
        # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 2, 0.0),
        # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (ngf * 2, 0.0),
        # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=4)

            output = tf.layers.conv3d_transpose(input, out_channels,
                                                **deconv_args)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=4)
        output = tf.layers.conv3d_transpose(input, 1, **deconv_args)
        layers.append(output)

    return tf.squeeze(layers[-1], -1)


def get_loss(input_images_placeholder, target_images_placeholder):
    """Calculates the loss from the logits and the labels.
    Args:

    Returns:
      loss: Loss tensor of type float.
    """

    loss = tf.reduce_mean(tf.abs(target_images_placeholder - input_images_placeholder))
    return loss



def get_training(loss, learning_rate):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.
    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(0.0002, 0.5)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
