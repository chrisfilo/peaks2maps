import os
import tensorflow as tf
# import trivial_model as model
# import models.unet as model
# import models.recnn as model
import models.meshnet as model
# import models.vae as model
from datasets import Peaks2MapsDataset
import datetime
import models.utils as utils

from tensorflow.contrib.learn.python.learn import learn_runner

tf.logging.set_verbosity(tf.logging.DEBUG)


def experiment_fn(run_config, params):
    """Train MNIST for a number of steps."""

    ds = Peaks2MapsDataset(target_shape=params.target_shape,
                           n_epochs=-1,
                           train_batch_size=params.batch_size,
                           validation_batch_size=1)

    estimator = utils.get_estimator(model.model_fn, run_config, params)

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,  # Estimator
        train_input_fn=ds.train_input_fn,  # First-class function
        eval_input_fn=ds.eval_input_fn,  # First-class function
        train_steps=params.train_steps,  # Minibatch steps
        eval_steps=None  # Use evaluation feeder until its empty
    )

    return experiment


if __name__ == '__main__':
    log_dir = "logs"
    current_run_subdir = os.path.join(
        "run_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    model_dir = os.path.join(log_dir, model.name, current_run_subdir)

    run_config = tf.contrib.learn.RunConfig(
        save_checkpoints_secs=120,
        model_dir=model_dir)

    params = tf.contrib.training.HParams(
        learning_rate=0.002,
        train_steps=2000000,
        target_shape=(32, 32, 32),
        batch_size=30,
        model_dir=model_dir,
        # images_to_plot=["137936_LANGUAGE.nii.gz", "138534_LANGUAGE.nii.gz",
        #                 "139233_LANGUAGE.nii.gz", "139637_LANGUAGE.nii.gz",
        #                 "140420_LANGUAGE.nii.gz"]
        images_to_plot=["100307_EMOTION.nii.gz", "100307_FACE-SHAPE.nii.gz",
                        "100307_GAMBLING.nii.gz", "100307_RELATIONAL.nii.gz",
                        "100307_SOCIAL.nii.gz"]
    )

    learn_runner.run(
        experiment_fn=experiment_fn,  # First-class function
        run_config=run_config,  # RunConfig
        schedule="train_and_evaluate",  # What to run
        hparams=params  # HParams
    )
