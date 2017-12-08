import tensorflow as tf
from functools import reduce
import math
import numpy as np

from .utils import get_evaluation_hooks, metric_fn

name = "pix2pix"

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

def model_fn(features, labels, mode, params):
    """Build the MNIST model up to where it may be used for inference.
    Args:

    Returns:

    """
    ngf = 64
    ndf = 64
    EPS = 1e-12
    beta1 = 0.5
    gan_weight = 1.0
    l1_weight = 1000.0
    lr = 0.0002
    layers = []

    training_flag = mode == tf.contrib.learn.ModeKeys.TRAIN

    labels, filenames = labels

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

    batchnorm_args = {"scale": True,
                      "gamma_initializer": tf.random_normal_initializer(1.0, 0.02),
                      "center": True,
                      "beta_initializer": tf.zeros_initializer(),
                      "name": "batchnorm"}

    def pad_and_conv(input, out_channels, conv_args):
        padded_input = tf.pad(input,
                              [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]],
                              mode="CONSTANT")
        convolved = tf.layers.conv3d(padded_input, out_channels,
                                     **conv_args)
        return convolved

    def create_generator(input):
        input = tf.expand_dims(input, -1)
        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("encoder_1"):
            output = pad_and_conv(input, ngf, conv_args)
            layers.append(output)

        layer_specs = [
            (ngf * 2, 0),
            # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            (ngf * 2, 0),
            # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            (ngf * 4, 0),
            #encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            (ngf * 8, 0),
            # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            # ngf * 8,
            # # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        ]

        for out_channels, dropout in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = pad_and_conv(layers[-1], out_channels, conv_args)
                output = tf.layers.batch_normalization(convolved, **batchnorm_args)
                if dropout > 0.0:
                    output = tf.layers.dropout(output, rate=dropout,
                                               training=training_flag)
                layers.append(output)

        layer_specs = [
            # (ngf * 8, 0.5),
            # # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (ngf * 8, 0.5),
            # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (ngf * 4, 0.5),
            # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (ngf * 2, 0),
            # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (ngf * 2, 0),
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
                output = tf.layers.batch_normalization(output,
                                                       training=training_flag,
                                                       **batchnorm_args)

                if dropout > 0.0:
                    output = tf.layers.dropout(output, rate=dropout,
                                               training=training_flag)

                layers.append(output)

        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            input = tf.concat([layers[-1], layers[0]], axis=4)
            output = tf.layers.conv3d_transpose(input, 1, **deconv_args)
            layers.append(output)

        predictions = tf.squeeze(layers[-1], -1)

        return predictions

    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 2
        layers = []

        discrim_inputs = tf.expand_dims(discrim_inputs, -1)
        discrim_targets = tf.expand_dims(discrim_targets, -1)
        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = pad_and_conv(input, ngf, conv_args)
            layers.append(convolved)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = ndf * min(2**(i+1), 8)
                #stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = pad_and_conv(layers[-1], out_channels, conv_args)
                normalized = tf.layers.batch_normalization(convolved,
                                                           training=training_flag,
                                                           **batchnorm_args)
                layers.append(normalized)

        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = tf.layers.conv3d(layers[-1], 1, strides=1,
                                         activation=tf.sigmoid,
                                         padding='same', kernel_size=4)
            layers.append(convolved)

        return layers[-1]

    with tf.variable_scope("generator"):
        predictions = create_generator(features)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(features, labels)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(features, predictions)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
        tf.summary.scalar("discrim_loss", discrim_loss)

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        tf.summary.scalar("gen_loss_GAN", gen_loss_GAN)
        gen_loss_L1 = tf.reduce_mean(tf.abs(labels - predictions))
        tf.summary.scalar("gen_loss_L1", gen_loss_L1)
        gen_loss = gen_loss_GAN * gan_weight + gen_loss_L1 * l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(lr, beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if
                         var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(lr, beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=gen_loss,
        # gen_loss_GAN=gen_loss_GAN,
        # gen_loss_L1=gen_loss_L1,
        # discrim_loss=discrim_loss,
        train_op=tf.group(update_losses, incr_global_step, gen_train),
        eval_metric_ops=metric_fn(tf.expand_dims(features, -1), labels,
                                  predictions),
        evaluation_hooks=get_evaluation_hooks(features, labels,
                                              predictions, filenames,
                                              mode, params)
    )
