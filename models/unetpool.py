import tensorflow as tf
from functools import reduce
import math
import numpy as np

from .utils import get_evaluation_hooks, metric_fn

name = "unetpool"


def model_fn(features, labels, mode, params):
    """Build the MNIST model up to where it may be used for inference.
    Args:

    Returns:

    """
    n_layers=3
    lowest_resolution = params.target_shape[0]
    for l in range(n_layers):
        lowest_resolution /= 2
    layers = np.arange(n_layers)

    input_images_placeholder = tf.expand_dims(features, -1)
    labels, filenames = labels

    prev_layer = input_images_placeholder
    encoding_layers = []
    for i in range(len(layers)):
        with tf.variable_scope("encoder_%d"%i):
            number_of_filters = lowest_resolution * 2 ** (layers[i])
            prev_layer = tf.layers.conv3d(prev_layer, number_of_filters,
                                          padding="same",
                                          activation=tf.nn.relu,
                                          kernel_size=5)
            encoding_layers.append(prev_layer)

            if i < len(layers) - 1:
                prev_layer = tf.layers.max_pooling3d(prev_layer,
                                                     pool_size=2,
                                                     strides=1)

    for i in range(1, len(layers)):
        with tf.variable_scope("decoder_%d" % i):
            print(i)
            number_of_filters = lowest_resolution * 2 ** (
                    len(layers) - layers[i] - 1)
            prev_layer = tf.layers.conv3d_transpose(prev_layer,
                                                    filters=number_of_filters,
                                                    kernel_size=5,
                                                    strides=2,
                                                    padding='same')
            prev_layer = tf.concat([prev_layer,
                                    encoding_layers[len(layers) - i - 1]],
                                   axis=4)

            prev_layer = tf.layers.conv3d(prev_layer, number_of_filters,
                                          padding="same",
                                          activation=tf.nn.relu,
                                          kernel_size=5)
            prev_layer = tf.layers.conv3d(prev_layer, number_of_filters,
                                          padding="same",
                                          activation=tf.nn.relu,
                                          kernel_size=5)

    prev_layer = tf.layers.conv3d(prev_layer, 1,
                                  padding="same",
                                  activation=tf.nn.tanh,
                                  kernel_size=1)

    predictions = tf.squeeze(prev_layer, -1)
    predictions = tf.Print(predictions, filenames, message="test")

    loss = tf.losses.mean_squared_error(labels, predictions)

    # Add a scalar summary for the snapshot loss.
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(0.0001)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = optimizer.minimize(loss,
                                      global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metric_fn(input_images_placeholder, labels,
                                  predictions),
        evaluation_hooks=get_evaluation_hooks(features, labels,
                                              predictions, filenames,
                                              mode, params)
    )




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
