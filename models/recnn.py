import tensorflow as tf
from functools import reduce
import math
import numpy as np

def get_inference(input_images_placeholder, input_shape, training_flag):
    """ https://hal.archives-ouvertes.fr/hal-01635455/document
    """
    n_layers = 15

    input_images_placeholder = tf.expand_dims(input_images_placeholder, -1)

    conv_args = {"strides": 1,
                 "kernel_size": 3,
                 "padding": "same",
                 "kernel_initializer": tf.contrib.layers.variance_scaling_initializer(factor = 2.0,
                                                                                      mode = 'FAN_IN',
                                                                                      uniform = False),
                 "bias_initializer": tf.initializers.zeros(),
                 "use_bias": True,
                 "name": "conv"}

    prev_layer = input_images_placeholder
    for i in range(n_layers-1):
        with tf.variable_scope("layer_%d" % i ):
            prev_layer = tf.layers.conv3d(prev_layer,
                                          filters=64,
                                          activation=tf.nn.leaky_relu,
                                          **conv_args)
    with tf.variable_scope("layer_%d" % (n_layers-1)):
        prev_layer = tf.layers.conv3d(prev_layer,
                                      filters=1,
                                      activation=None,
                                      **conv_args)

    return tf.squeeze(prev_layer, -1)


def get_loss(input_images_placeholder, target_images_placeholder):
    """Calculates the loss from the logits and the labels.
    Args:

    Returns:
      loss: Loss tensor of type float.
    """

    loss = tf.losses.mean_squared_error(target_images_placeholder, input_images_placeholder)
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
    optimizer = tf.train.AdamOptimizer(0.001, 0.5)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
