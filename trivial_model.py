import tensorflow as tf
from functools import reduce
import math


def get_inference(input_images_placeholder, input_shape):
    """Build the MNIST model up to where it may be used for inference.
    Args:

    Returns:

    """
    # Hidden 1
    with tf.name_scope('hidden1'):
        voxel_count = reduce(lambda x, y: x * y, input_shape)
        weights = tf.Variable(
            tf.truncated_normal(input_shape, mean=0.0,
                                stddev=0.1),
            name='weights')
        hidden1 = tf.multiply(input_images_placeholder, weights)
    return hidden1


def get_loss(input_images_placeholder, target_images_placeholder):
    """Calculates the loss from the logits and the labels.
    Args:

    Returns:
      loss: Loss tensor of type float.
    """
    loss = tf.reduce_mean(
        tf.abs(target_images_placeholder - input_images_placeholder),
        name='abs_diff_mean')
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
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
