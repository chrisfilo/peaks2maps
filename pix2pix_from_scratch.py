import os
import tensorflow as tf
# import trivial_model as model
#import models.unet2 as model
import models.unet as model
#import models.pix2pix as model
#import models.unetpool as model
#import models.recnn as model
from tensorflow.python.estimator.run_config import RunConfig

#import models.meshnet as model
# import models.vae as model
# import models.trivial_conv as model
#import models.fixed_conv as model
from datasets import Peaks2MapsDataset
import datetime
import models.utils as utils

from tensorflow.contrib.learn.python.learn import learn_runner

tf.logging.set_verbosity(tf.logging.INFO)


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
    model_dir = os.path.join(log_dir, model.name, "11_aug")# current_run_subdir)

    session_config = tf.ConfigProto()
    run_config = tf.contrib.learn.RunConfig(
        save_checkpoints_secs=600,
        model_dir=model_dir,
        save_summary_steps=100,
        log_step_count_steps=100,
        session_config=session_config)

    params = tf.contrib.training.HParams(
        learning_rate=0.0002,
        train_steps=2000000,
        target_shape=(32, 32, 32),
        batch_size=40,
        model_dir=model_dir,
        # images_to_plot=["137936_LANGUAGE.nii.gz", "138534_LANGUAGE.nii.gz",
        #                 "139233_LANGUAGE.nii.gz", "139637_LANGUAGE.nii.gz",
        #                 "140420_LANGUAGE.nii.gz"]
        # images_to_plot=["100307_EMOTION.nii.gz", "100307_FACE-SHAPE.nii.gz",
        #                 "100307_GAMBLING.nii.gz", "100307_RELATIONAL.nii.gz",
        #                 "100307_SOCIAL.nii.gz", "100307_LANGUAGE.nii.gz"],
        images_to_plot=["5494.nii.gz", "10263.nii.gz",
                          "8288.nii.gz", "31492.nii.gz",
                          "32461.nii.gz", "37992.nii.gz", "l_100307_EMOTION.nii.gz",
                         "l_100307_FACE-SHAPE.nii.gz",
                        "l_100307_GAMBLING.nii.gz", "l_100307_RELATIONAL.nii.gz",
                        "l_100307_SOCIAL.nii.gz", "l_100307_LANGUAGE.nii.gz"]
    )

    learn_runner.run(
        experiment_fn=experiment_fn,  # First-class function
        run_config=run_config,  # RunConfig
        schedule="train_and_evaluate",  # What to run
        hparams=params  # HParams
    )
