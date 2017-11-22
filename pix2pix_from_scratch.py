import time
import os
import tensorflow as tf
#import trivial_model as model
import models.unet as model
from datasets import Peaks2MapsDataset
import numpy as np
import datetime

learning_rate = 0.01
log_dir = 'logs'
max_steps = 2000


def run_training():
    """Train MNIST for a number of steps."""

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():

        ds = Peaks2MapsDataset(target_shape=(32, 32, 32),
                               n_epochs=10000,
                               train_batch_size=20,
                               validation_batch_size=1)
        # Generate placeholders for the images and labels.
        #input_images, target_images, input_shape, handle, training_iterator, validation_iterator = get_data(batch_size)


        training_flag = tf.placeholder(tf.bool, name="training_flag")

        # Build a Graph that computes predictions from the inference model.
        inference_model = model.get_inference(ds.input_image,
                                              ds.target_shape,
                                              training_flag)

        # Add to the Graph the Ops for loss calculation.
        loss = model.get_loss(inference_model, ds.target_image)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = model.get_training(loss, learning_rate)


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
        current_run_subdir = os.path.join("run_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        train_summary_writer = tf.summary.FileWriter(os.path.join(log_dir,
                                                                  current_run_subdir,
                                                                  "train"),
                                                                  sess.graph)
        test_summary_writer = tf.summary.FileWriter(os.path.join(log_dir,
                                                                 current_run_subdir,
                                                                 "test"))

        mean_loss_placeholder = tf.placeholder(tf.float32, shape=())
        mean_loss_summary = tf.summary.scalar('loss', mean_loss_placeholder)
        losses_placeholder = tf.placeholder(tf.float32, shape=(781))
        losses_histogram_summary = tf.summary.histogram('loss_histogram',
                                                        losses_placeholder)
        images_count = 5
        summary_images_input = [ds.get_plot_op(ds.input_image, 'example_%02d_input'%i) for i in range(images_count)]
        summary_images_output = [ds.get_plot_op(inference_model, 'example_%02d_output'%i) for i in range(images_count)]
        summary_images_target = [ds.get_plot_op(ds.target_image, 'example_%02d_target'%i) for i in range(images_count)]

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        training_handle = sess.run(ds.training_iterator.string_handle())
        validation_handle = sess.run(ds.validation_iterator.string_handle())

        # Start the training loop.

        print("Begin training model with %d parameters" % np.sum(
            [np.prod(v.shape) for v in tf.trainable_variables()]))
        step = 0

        sess.run(ds.validation_iterator.initializer)
        while True:
            try:
                sess.run(ds.input_image, feed_dict={
                    ds.handle: validation_handle})
            except tf.errors.OutOfRangeError:
                break
        sess.run(ds.validation_iterator.initializer)

        while True:

            try:
                start_time = time.time()

                # Run one step of the model.  The return values are the activations
                # from the `train_op` (which is discarded) and the `loss` Op.  To
                # inspect the values of your Ops or variables, you may include them
                # in the list passed to sess.run() and the value tensors will be
                # returned in the tuple from the call.
                _, loss_value = sess.run([train_op, loss],
                                         feed_dict={ds.handle: training_handle,
                                                    training_flag: True})

                duration = time.time() - start_time

                # Write the summaries and print an overview fairly often.
                if step % 100 == 0:
                    print('Step %d: loss = %.2f (%.3f sec)' % (
                        step, loss_value, duration))
                    #summary_str = sess.run(summary, feed_dict={ds.handle: training_handle})
                    sum_mean_loss = sess.run(mean_loss_summary, feed_dict={
                        mean_loss_placeholder: loss_value,
                        ds.handle: validation_handle})
                    train_summary_writer.add_summary(sum_mean_loss, step)
                    #train_summary_writer.add_summary(summary_str, step)
                    train_summary_writer.flush()

                    sess.run(ds.validation_iterator.initializer)
                    if step == 0:
                        for i in range(images_count):
                            sum_img_input, sum_img_target = sess.run([summary_images_input[i], summary_images_target[i]],
                                feed_dict={
                                    ds.handle: validation_handle})
                            test_summary_writer.add_summary(sum_img_target, step)
                            test_summary_writer.add_summary(sum_img_input, step)
                        sess.run(ds.validation_iterator.initializer)

                    losses = []
                    while True:
                        try:
                            if len(losses) in range(images_count):
                                cur_loss, sum_img = sess.run(
                                        [loss, summary_images_output[len(losses)]],
                                        feed_dict={
                                            ds.handle: validation_handle,
                                            training_flag: False})
                                test_summary_writer.add_summary(sum_img, step)
                            else:
                                cur_loss = sess.run(loss, feed_dict={
                                    ds.handle: validation_handle,
                                    training_flag: False})

                            losses.append(cur_loss)
                        except tf.errors.OutOfRangeError:
                            break

                    losses = np.array(losses)

                    sum_mean_loss = sess.run(mean_loss_summary, feed_dict={mean_loss_placeholder: np.mean(losses),
                                                                           ds.handle: validation_handle})
                    sum_loss_dist = sess.run(losses_histogram_summary, feed_dict={losses_placeholder: losses,
                                                                                  ds.handle: validation_handle})
                    test_summary_writer.add_summary(sum_mean_loss, step)
                    test_summary_writer.add_summary(sum_loss_dist, step)
                    test_summary_writer.flush()


            except tf.errors.OutOfRangeError:
                break
            step += 1





if __name__ == '__main__':
    run_training()
