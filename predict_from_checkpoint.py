import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)

import numpy.linalg as npl
import nibabel as nb
import numpy as np
import models.unet as model
from datasets import Peaks2MapsDataset, _get_data, save_nii

def _get_resize_arg(target_shape):
    mni_shape_mm = np.array([148.0, 184.0, 156.0])
    target_resolution_mm = np.ceil(
        mni_shape_mm / np.array(target_shape)).astype(
        np.int32)
    target_affine = np.array([[4., 0., 0., -75.],
                              [0., 4., 0., -105.],
                              [0., 0., 4., -70.],
                              [0., 0., 0., 1.]])
    target_affine[0, 0] = target_resolution_mm[0]
    target_affine[1, 1] = target_resolution_mm[1]
    target_affine[2, 2] = target_resolution_mm[2]
    return target_affine, list(target_shape)

target_shape = (32, 32, 32)



# signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
# input_key = 'input'
# output_key = 'output'
#
# print("loading graph")
# export_path = 'G:/My Drive/data/peaks2maps/saved_models/1527885839/'
# meta_graph_def = tf.saved_model.loader.load(sess,
#                                             [tf.saved_model.tag_constants.SERVING],
#                                             export_path)
# signature = meta_graph_def.signature_def
#
# x_tensor_name = signature[signature_key].inputs[input_key].name
# y_tensor_name = signature[signature_key].outputs[output_key].name
#
# print("getting endpoints")
# x = sess.graph.get_tensor_by_name(x_tensor_name)
# y = sess.graph.get_tensor_by_name(y_tensor_name)


real_pt = (75, 24 , 30)
affine, _ = _get_resize_arg(target_shape)
vox_pt = np.rint(nb.affines.apply_affine(npl.inv(affine), real_pt)).astype(int)
encoded_coords = np.zeros([1,] + list(target_shape))
encoded_coords[0, vox_pt[0],vox_pt[1], vox_pt[2]] = 1

model_dir='G:/My Drive/data/peaks2maps/saved_models/ohbm2018_model'

params = tf.contrib.training.HParams(
    learning_rate=0.0002,
    train_steps=2000000,
    target_shape=target_shape,
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

model = tf.estimator.Estimator(model.model_fn, model_dir=model_dir,
                               params=params)

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 8
config.inter_op_parallelism_threads = 8

ds = Peaks2MapsDataset(target_shape=(32, 32, 32),
                       n_epochs=1,
                       train_batch_size=1,
                       validation_batch_size=1)


contrast_list = [[[60, -39, 13]]]

def generate_input_fn():
    def get_generator():
        affine, _ = _get_resize_arg(target_shape)
        for contrast in contrast_list:
            encoded_coords = np.zeros(list(target_shape))
            for real_pt in contrast:
                vox_pt = np.rint(nb.affines.apply_affine(npl.inv(affine), real_pt)).astype(int)
                encoded_coords[vox_pt[0], vox_pt[1], vox_pt[2]] = 1
            yield (encoded_coords, encoded_coords)

    dataset = tf.data.Dataset.from_generator(get_generator,
                                             (tf.float32, tf.float32),
                                             (tf.TensorShape(target_shape), tf.TensorShape(target_shape)))
    dataset = dataset.batch(1)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()




with tf.Session(config=config) as sess:
    print("begin inference")
    results = model.predict(generate_input_fn)
    for result in results:
        save_nii(result, (32, 32, 32),
                 "C:/scratch/predicted_map.nii.gz")

print("begin inference")
# y_out = sess.run(y, {x: encoded_coords})
# print(y_out)