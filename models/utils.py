import os
import tensorflow as tf
from datasets import get_plot_op


def metric_fn(labels, predictions):
    return {
        "pearson_r": tf.contrib.metrics.streaming_pearson_correlation(labels,
                                                                      predictions),
        "mean_absolute_error": tf.contrib.metrics.streaming_mean_absolute_error(
            labels, predictions),
        "mean_squared_error": tf.contrib.metrics.streaming_mean_squared_error(
            labels, predictions)}


def get_evaluation_hooks(features, labels, predictions, filenames, mode,
                         params):
    if mode == tf.contrib.learn.ModeKeys.EVAL:
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
    return evaluation_hooks


def get_estimator(model_fn, run_config, params):
    return tf.estimator.Estimator(model_fn=model_fn,  # First-class function
                                  params=params,  # HParams
                                  config=run_config  # RunConfig
                                  )
