import tensorflow as tf

from datasets import Peaks2MapsDataset, _get_data, save_nii
import models.unet as model
import models.fixed_conv as reference_model
import os, datetime

log_dir = "logs"
current_run_subdir = os.path.join(
    "run_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
model_dir = os.path.join(log_dir, model.name,
                         "3_nohcp")  # current_run_subdir)

run_config = tf.contrib.learn.RunConfig(
    save_checkpoints_secs=1200,
    model_dir=model_dir)

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
                    "32461.nii.gz", "37992.nii.gz"]
)

with tf.Session() as sess:
    validation_dataset, validation_shape = _get_data(8, 1,
                                                          "D:/data/hcp_statmaps/val_all_tasks",
                                                          1,
                                                          'D:/data/hcp_statmaps/val_no_language_unseen_partitipants',
                                                          False,
                                                          (32, 32, 32),
                                                          False)

    validation_iterator = validation_dataset.make_one_shot_iterator()
    filenames = []
    while True:
        try:
            features, (labels, filename) = sess.run(validation_iterator.get_next())
            filename = filename[0][0].decode('utf-8')
            save_nii(features, (32, 32, 32), "D:/data/peaks2maps/validation3/features/f_" + filename)
            save_nii(labels, (32, 32, 32),
                     "D:/data/peaks2maps/validation3/labels/l_" + filename)
            print(filename)
            filenames.append(filename)
        except tf.errors.OutOfRangeError:
            break
    print(len(filenames))

ds = Peaks2MapsDataset(target_shape=(32, 32, 32),
                       n_epochs=1,
                       train_batch_size=1,
                       validation_batch_size=1)

eval_input_fn = ds.eval_input_fn

unet = tf.estimator.Estimator(model.model_fn, model_dir='logs/unet/3_nohcp/',
                              params=params)

predictions = unet.predict(input_fn=eval_input_fn)

for i, prediction in enumerate(predictions):
    save_nii(prediction, (32, 32, 32),
             "D:/data/peaks2maps/validation3/predictions/p_" + filenames[i])
    print(i)


ds = Peaks2MapsDataset(target_shape=(32, 32, 32),
                       n_epochs=1,
                       train_batch_size=1,
                       validation_batch_size=1)

eval_input_fn = ds.eval_input_fn

unet = tf.estimator.Estimator(reference_model.model_fn, model_dir='logs/fixed_conv/run_2017-12-04_19-44-26',
                              params=params)

predictions = unet.predict(input_fn=eval_input_fn)

for i, prediction in enumerate(predictions):
    save_nii(prediction, (32, 32, 32),
             "D:/data/peaks2maps/validation3/predictions_ref/p_" + filenames[i])
    print(i)