import tensorflow as tf
from functools import reduce
import math
import numpy as np

name = "vae"

def get_inference(input_images_placeholder, target_images_placeholder, training_flag):
    """Build the MNIST model up to where it may be used for inference.
    Args:

    Returns:

    """
    ngf = 32
    z_len = 100

    input_images_placeholder = tf.expand_dims(input_images_placeholder, -1)

    conv_args = {"strides": 2,
                 "kernel_size": 4,
                 "activation": tf.nn.leaky_relu,
                 "kernel_initializer": tf.random_normal_initializer(0, 0.02),
                 "name": "conv",
                 "padding": "same",
                 "use_bias": False}

    deconv_args = conv_args.copy()
    deconv_args["name"] = "deconv"

    batchnorm_args = {"scale": True,
                      "gamma_initializer": tf.random_normal_initializer(1.0, 0.02),
                      "center": True,
                      "beta_initializer": tf.zeros_initializer(),
                      "name": "batchnorm"}

    layer_specs = [
        (ngf, 0.0),
        (ngf * 2, 0.0),
        (ngf * 2, 0.0),
        (ngf * 4, 0.0),
        (ngf * 8, 0.0)
    ]

    prev_layer = input_images_placeholder
    for i, (out_channels, dropout) in enumerate(layer_specs):
        with tf.variable_scope("encoder_%d" % i) :
            prev_layer = tf.layers.conv3d(prev_layer, out_channels,
                                          **conv_args)
            # prev_layer = tf.layers.batch_normalization(prev_layer,
            #                                            training=training_flag,
            #                                            **batchnorm_args)
            prev_layer_shape = tf.shape(prev_layer)
            if dropout > 0.0:
                prev_layer = tf.layers.dropout(prev_layer, rate=dropout,
                                               training=training_flag,
                                               noise_shape=[prev_layer_shape[0], 1, 1, 1, prev_layer_shape[4]])

    prev_layer = tf.contrib.layers.flatten(prev_layer)
    latent_mean = tf.layers.dense(prev_layer, z_len, name="latent_mean",
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                  bias_initializer=tf.constant_initializer(0.0))
    latent_stdev = tf.layers.dense(prev_layer, z_len, name="latent_stdev",
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                   bias_initializer=tf.constant_initializer(0.0))
    samples = tf.random_normal(tf.shape(latent_mean), 0, 1,
                               dtype=tf.float32)
    guessed_z = latent_mean + (latent_stdev * samples)
    prev_layer = tf.layers.dense(guessed_z, ngf * 8,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                 bias_initializer=tf.constant_initializer(0.0))
    prev_layer = tf.reshape(prev_layer, prev_layer_shape)
    prev_layer = tf.nn.relu(prev_layer)

    layer_specs = [
        (ngf * 8, 0.5),
        (ngf * 4, 0.5),
        (ngf * 2, 0.0),
        (ngf * 2, 0.0),
        (1, 0.0)
    ]

    for i, (out_channels, dropout) in enumerate(layer_specs):
        with tf.variable_scope("decoder_%d" % i):
            prev_layer = tf.layers.conv3d_transpose(prev_layer, out_channels,
                                                    **deconv_args)
            # prev_layer = tf.layers.batch_normalization(prev_layer,
            #                                            training=training_flag,
            #                                            **batchnorm_args)
            if dropout > 0.0:
                prev_layer_shape = tf.shape(prev_layer)
                prev_layer = tf.layers.dropout(prev_layer, rate=dropout,
                                               training=training_flag,
                                               noise_shape=[prev_layer_shape[0], 1, 1, 1, prev_layer_shape[4]])

    prediction = tf.squeeze(prev_layer, -1)

    reconstruction_loss = tf.losses.absolute_difference(target_images_placeholder, prediction)
    # target_flat = tf.contrib.layers.flatten(target_images_placeholder)
    # pred_flat = tf.contrib.layers.flatten(prediction)
    # reconstruction_loss = -tf.reduce_sum(target_flat * tf.log(1e-8 + pred_flat) + (1 - target_flat) * tf.log(1e-8 + 1 - pred_flat), 1)

    latent_loss = tf.reduce_mean(0.5 * tf.reduce_sum(
        tf.square(latent_mean) + tf.square(latent_stdev) - tf.log(
            tf.square(latent_stdev)) - 1, 1))
    cost = reconstruction_loss + 0.0001*latent_loss
    # cost = reconstruction_loss

    optimizer = tf.train.AdamOptimizer(0.001)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op, prediction, latent_loss


def get_loss(input_images_placeholder, target_images_placeholder):
    """Calculates the loss from the logits and the labels.
    Args:

    Returns:
      loss: Loss tensor of type float.
    """

    loss = tf.losses.absolute_difference(target_images_placeholder, input_images_placeholder)
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
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
