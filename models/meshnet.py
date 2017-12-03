import tensorflow as tf
from functools import reduce
import math
import numpy as np
from tensorflow.contrib.learn import ModeKeys
import os
from datasets import get_plot_op


def metric_fn(labels, predictions):
    return { "pearson_r": tf.contrib.metrics.streaming_pearson_correlation(labels, predictions),
             "mean_absolute_error": tf.contrib.metrics.streaming_mean_absolute_error(
                 labels, predictions),
             "mean_squared_error": tf.contrib.metrics.streaming_mean_squared_error(
                 labels, predictions)}


def model_fn(features, labels, mode, params):
    """ https://github.com/Entodi/MeshNet/blob/master/models/nodp_model.lua
    https://arxiv.org/pdf/1711.00457.pdf
    http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7966333
    """
    n_layers = 7

    n_filters = [21, 21, 21, 21, 21, 21, 1]

    kernel_sizes = [3, 3, 3, 3, 3, 3, 1]

    dilations = [1, 1, 2, 4, 8, 1, 1]

    activation_functions = [tf.nn.relu]*6 + [tf.nn.tanh]

    input_images_placeholder = tf.expand_dims(features, -1)

    is_training = mode == ModeKeys.TRAIN

    labels, filenames = labels

    batchnorm_args = {
        # "scale": True,
        #               "gamma_initializer": tf.random_normal_initializer(1.0, 0.02),
        #               "center": True,
        #               "beta_initializer": tf.zeros_initializer(),
                      "name": "batchnorm"}

    conv_args = {"strides": 1,
                 "padding": "same",
                 "kernel_initializer": tf.contrib.layers.variance_scaling_initializer(factor = 2.0,
                                                                                      mode = 'FAN_IN',
                                                                                      uniform = False),
                 # "bias_initializer": None,
                 # "use_bias": False,
                 "name": "conv"}

    prev_layer = input_images_placeholder
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % i ):
            prev_layer = tf.layers.conv3d(prev_layer,
                                          filters=n_filters[i],
                                          activation=None,
                                          kernel_size=kernel_sizes[i],
                                          dilation_rate=dilations[i],
                                          **conv_args)
            # prev_layer = tf.layers.batch_normalization(prev_layer,
            #                                            training=is_training,
            #                                            **batchnorm_args)
            prev_layer = activation_functions[i](prev_layer)

    predictions = tf.squeeze(prev_layer, -1)

    if mode == ModeKeys.EVAL:
        summaries = []
        for image_to_plot in params.images_to_plot:
            is_to_plot = tf.equal(tf.squeeze(filenames), image_to_plot)
            is_first_eval = tf.equal(tf.train.get_global_step(), 1)
            plot_references = tf.logical_and(is_to_plot, is_first_eval)
            image_to_plot = image_to_plot.replace('.nii.gz', '')

            summary = tf.cond(plot_references,
                              lambda: get_plot_op(features,
                                                  params.target_shape,
                                                  'feature'),
                              lambda: tf.summary.histogram("ignore_me", [0]),
                              name="%s_0feature" % image_to_plot)
            summaries.append(summary)

            summary = tf.cond(is_to_plot,
                              lambda: get_plot_op(predictions,
                                                  params.target_shape,
                                                  'prediction'),
                              lambda: tf.summary.histogram("ignore_me", [0]),
                              name="%s_1prediction" % image_to_plot)
            summaries.append(summary)

            summary = tf.cond(plot_references,
                              lambda: get_plot_op(labels,
                                                  params.target_shape,
                                                  'label'),
                              lambda: tf.summary.histogram("ignore_me", [0]),
                              name="%s_2label" % image_to_plot)
            summaries.append(summary)

        evaluation_hooks = [tf.train.SummarySaverHook(
            save_steps=1,
            output_dir=os.path.join(params.model_dir, "eval"),
            summary_op=tf.summary.merge(summaries))]
    else:
        evaluation_hooks = None


    predictions = tf.Print(predictions, filenames, message="test")



    loss = tf.losses.absolute_difference(labels, predictions)

    # Add a scalar summary for the snapshot loss.
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(0.001, 0.5)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metric_fn(labels, predictions),
            evaluation_hooks=evaluation_hooks
        )


name = "meshnet"



def get_estimator(run_config, params):

    return tf.estimator.Estimator(model_fn=model_fn,  # First-class function
                                   params=params,  # HParams
                                   config=run_config  # RunConfig
                )
