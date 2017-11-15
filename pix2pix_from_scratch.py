import time
import os
import tensorflow as tf
import trivial_model as model
from datasets import get_data, gen_plot
import numpy as np

batch_size = 10
learning_rate = 0.01
log_dir = 'test_log'
max_steps = 2000


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      images_placeholder: The images placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    input_images_placeholder, target_images_placeholder, input_shape = get_data(
        1)

    inference_model = model.get_inference(input_images_placeholder,
                                          input_shape)
    eval = tf.contrib.metrics.streaming_pearson_correlation(inference_model,
                                                            target_images_placeholder)
    corrs = []
    while True:
        try:
            feed_dict = data_set.fill_feed_dict(images_placeholder,
                                                labels_placeholder)
            corrs.append(sess.run(eval, feed_dict=feed_dict))
        except tf.errors.OutOfRangeError:
            break

    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))

def run_training():
    """Train MNIST for a number of steps."""

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        input_images, target_images, input_shape, handle, training_iterator, validation_iterator = get_data(batch_size)

        # Build a Graph that computes predictions from the inference model.
        inference_model = model.get_inference(input_images,
                                              input_shape)

        # Add to the Graph the Ops for loss calculation.
        loss = model.get_loss(inference_model, target_images)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = model.get_training(loss, learning_rate)

        eval = tf.contrib.metrics.streaming_pearson_correlation(
            inference_model,
            target_images)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.group(tf.global_variables_initializer(),
                 tf.local_variables_initializer())

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        train_summary_writer = tf.summary.FileWriter(os.path.join(log_dir, "train"),
                                                                  sess.graph)
        test_summary_writer = tf.summary.FileWriter(os.path.join(log_dir, "test"),
            sess.graph)

        mean_correlation_ = tf.placeholder(tf.float32, shape=())
        mean_correlation_summary = tf.summary.scalar('mean_correlation', mean_correlation_)
        correlations_ = tf.placeholder(tf.float32, shape=(781))
        correlation_histogram_summary = tf.summary.histogram('correlation_histogram',
                                                             correlations_)

        summary_image_output = summary_glass_brain(inference_model, 'example_1_output')
        summary_image_target = summary_glass_brain(target_images, 'example_1_target')

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())

        # Start the training loop.
        step = 0
        while True:
            try:
                start_time = time.time()

                # Run one step of the model.  The return values are the activations
                # from the `train_op` (which is discarded) and the `loss` Op.  To
                # inspect the values of your Ops or variables, you may include them
                # in the list passed to sess.run() and the value tensors will be
                # returned in the tuple from the call.
                _, loss_value = sess.run([train_op, loss],
                                         feed_dict={handle: training_handle})

                duration = time.time() - start_time

                # Write the summaries and print an overview fairly often.
                if step % 100 == 0:
                    print('Step %d: loss = %.2f (%.3f sec)' % (
                        step, loss_value, duration))
                    summary_str = sess.run(summary, feed_dict={handle: training_handle})
                    train_summary_writer.add_summary(summary_str, step)
                    train_summary_writer.flush()

                    sess.run(validation_iterator.initializer)
                    if step == 0:
                        sum_img_trg = sess.run(summary_image_target,
                            feed_dict={
                                handle: validation_handle})
                        test_summary_writer.add_summary(sum_img_trg, step)
                        sess.run(validation_iterator.initializer)
                    corrs = []
                    while True:
                        try:
                            if len(corrs) == 0:
                                corr, sum_img = sess.run(
                                        [eval, summary_image_output],
                                        feed_dict={
                                            handle: validation_handle})
                            else:
                                corr = sess.run(eval, feed_dict={
                                    handle: validation_handle})
                            corrs.append(corr[0])
                        except tf.errors.OutOfRangeError:
                            break
                    corrs = np.array(corrs)

                    sum = sess.run(mean_correlation_summary, feed_dict={mean_correlation_: np.nanmean(corrs),
                                                                  handle: validation_handle})
                    # sess.run(correlation_histogram_summary, feed_dict={correlations_: corrs,
                    #                                                    handle: validation_handle})
                    test_summary_writer.add_summary(sum, step)
                    # Add image summary
                    test_summary_writer.add_summary(sum_img, step)
                    test_summary_writer.flush()

            except tf.errors.OutOfRangeError:
                break
            step += 1


def summary_glass_brain(inference_model, label):
    plot = tf.py_func(gen_plot, [inference_model], tf.string)
    image = tf.image.decode_png(plot)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    summary_image = tf.summary.image(label,
                                     image, max_outputs=200000)
    return summary_image


if __name__ == '__main__':
    run_training()