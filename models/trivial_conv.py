import tensorflow as tf
from .utils import get_evaluation_hooks, metric_fn

name = "trivial_conv"


def model_fn(features, labels, mode, params):
    """ single layer single filter convolution
    """
    input_images_placeholder = tf.expand_dims(features, -1)

    labels, filenames = labels

    prev_layer = tf.layers.conv3d(input_images_placeholder,
                                  filters=1,
                                  strides=1,
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                      factor=2.0,
                                      mode='FAN_IN',
                                      uniform=False),
                                  activation=None,
                                  kernel_size=31,
                                  bias_initializer=None,
                                  use_bias=False,
                                  padding='same')

    predictions = tf.squeeze(prev_layer, -1)

    #predictions = tf.Print(predictions, filenames, message="test")

    loss = tf.losses.mean_squared_error(labels, predictions)

    # Add a scalar summary for the snapshot loss.
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
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
