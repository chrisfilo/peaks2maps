import tensorflow as tf
import numpy as np
from .utils import get_evaluation_hooks, metric_fn, gkern

name = "fixed_conv"


def model_fn(features, labels, mode, params):
    """ single layer single fixed filter convolution - no training
    """
    input_images_placeholder = tf.expand_dims(features, -1)

    labels, filenames = labels

    kern = gkern(params.target_shape)

    prev_layer = tf.layers.conv3d(input_images_placeholder,
                                  filters=1,
                                  strides=1,
                                  kernel_initializer=tf.constant_initializer(kern),
                                  activation=None,
                                  kernel_size=5,
                                  bias_initializer=None,
                                  use_bias=False,
                                  padding='same',
                                  trainable=False)

    predictions = tf.squeeze(prev_layer, -1)

    #predictions = tf.Print(predictions, filenames, message="test")

    loss = tf.losses.mean_squared_error(labels, predictions)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=tf.no_op(),
        eval_metric_ops=metric_fn(input_images_placeholder, labels,
                                  predictions),
        evaluation_hooks=get_evaluation_hooks(features, labels,
                                              predictions, filenames,
                                              mode, params)
    )
