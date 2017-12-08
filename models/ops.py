import tensorflow as tf

def dropout3d(input, rate, training):
    shape = tf.shape(input)
    return tf.layers.dropout(input, rate=rate, training=training,
                             noise_shape=[shape[0], 1, 1, 1, shape[4]])