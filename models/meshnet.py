import tensorflow as tf
from .utils import get_evaluation_hooks, metric_fn


def model_fn(features, labels, mode, params):
    """ https://github.com/Entodi/MeshNet/blob/master/models/nodp_model.lua
    https://arxiv.org/pdf/1711.00457.pdf
    http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7966333
    """
    n_layers = 7

    n_filters = [21, 21, 21, 21, 21, 21, 1]

    kernel_sizes = [3, 3, 3, 3, 3, 3, 1]

    dilations = [1, 1, 2, 4, 8, 1, 1]

    activation_functions = [tf.nn.relu] * 6 + [tf.nn.tanh]

    input_images_placeholder = tf.expand_dims(features, -1)

    labels, filenames = labels

    batchnorm_args = {
        # "scale": True,
        #               "gamma_initializer": tf.random_normal_initializer(1.0, 0.02),
        #               "center": True,
        #               "beta_initializer": tf.zeros_initializer(),
        "name": "batchnorm"}

    conv_args = {"strides": 1,
                 "padding": "same",
                 "kernel_initializer": tf.contrib.layers.variance_scaling_initializer(
                     factor=2.0,
                     mode='FAN_IN',
                     uniform=False),
                 # "bias_initializer": None,
                 # "use_bias": False,
                 "name": "conv"}

    prev_layer = input_images_placeholder
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % i):
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

    #predictions = tf.Print(predictions, filenames, message="test")

    loss = tf.losses.mean_squared_error(labels, predictions)

    # Add a scalar summary for the snapshot loss.
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(params.learning_rate, 0.5)
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
        eval_metric_ops=metric_fn(labels, predictions),
        evaluation_hooks=get_evaluation_hooks(features, labels,
                                              predictions, filenames,
                                              mode, params)
    )


name = "meshnet"
