import matplotlib as mpl
mpl.use('agg')
from glob import glob
import tensorflow as tf
import os
import nibabel as nb
import numpy as np
from scipy.ndimage.interpolation import zoom
from nilearn import image

# class Peaks2MapsDataset:
#
#     def __init__(self, conform_shape=(64, 64, 64)):
#         self.conform_shape = conform_shape
#
#     def fill_feed_dict(data_set, images_pl, labels_pl):
#         """Fills the feed_dict for training the given step.
#         A feed_dict takes the form of:
#         feed_dict = {
#             <placeholder>: <tensor of values to be passed for placeholder>,
#             ....
#         }
#         Args:
#           data_set: The set of images and labels, from input_data.read_data_sets()
#           images_pl: The images placeholder, from placeholder_inputs().
#           labels_pl: The labels placeholder, from placeholder_inputs().
#         Returns:
#           feed_dict: The feed dictionary mapping from placeholders to values.
#         """
#         # Create the feed_dict for the placeholders filled with the next
#         # `batch size` examples.
#         images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
#                                                        FLAGS.fake_data)
#         feed_dict = {
#             images_pl: images_feed,
#             labels_pl: labels_feed,
#         }
#         return feed_dict



def get_data(batch_size, nthreads=8):
    training_dataset, target_shape = _get_data(nthreads, batch_size,
                                               "D:/data/hcp_statmaps/train", 100,
                                               'D:/drive/workspace/peaks2maps/cache_train',
                                               True)
    validation_dataset, target_shape = _get_data(nthreads, 1,
                                                 "D:/data/hcp_statmaps/val", 1,
                                                 'D:/drive/workspace/peaks2maps/cache_val',
                                                 False)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, training_dataset.output_types, training_dataset.output_shapes)
    input, target = iterator.get_next()

    # You can use feedable iterators with a variety of different kinds of iterator
    # (such as one-shot and initializable iterators).
    training_iterator = training_dataset.make_one_shot_iterator()
    validation_iterator = validation_dataset.make_initializable_iterator()

    return input, target, target_shape, handle, training_iterator, validation_iterator


def _get_data(nthreads, batch_size, src_folder, n_epochs, cache, shuffle):
    filenames = tf.constant(glob(os.path.join(src_folder, "*.nii.gz")))
    dataset = tf.data.Dataset.from_tensor_slices((filenames,))

    def _get_resize_arg(target_resolution_mm):
        mni_shape_mm = np.array([148.0, 184.0, 156.0])
        target_shape = np.ceil(mni_shape_mm / target_resolution_mm).astype(
            np.int32)
        target_affine = np.array([[4., 0., 0., -75.],
                                  [0., 4., 0., -105.],
                                  [0., 0., 4., -70.],
                                  [0., 0., 0., 1.]])
        target_affine[0, 0] = target_affine[1, 1] = target_affine[
            2, 2] = target_resolution_mm
        return target_affine, list(target_shape)

    target_affine, target_shape = _get_resize_arg(4.0)

    def _read_py_function(filename, target_affine, target_shape):
        nii = image.resample_img(
            filename.decode('utf-8'),
            target_affine=target_affine, target_shape=target_shape)
        data = nii.get_data()
        data = data.astype(np.float32)
        return data

    target_affine_tf = tf.constant(target_affine)
    target_shape_tf = tf.constant(target_shape)
    dataset = dataset.map(lambda filename: tuple(tf.py_func(_read_py_function,
                                                            [filename,
                                                             target_affine_tf,
                                                             target_shape_tf],
                                                            [tf.float32])),
                          num_parallel_calls=nthreads)

    def _resize(data):
        return tf.reshape(data, target_shape)

    dataset = dataset.map(_resize)
    # def _smooth_py_function(data):
    #     target_shape = np.array([64, 64, 64])
    #     zoom_factors = target_shape/data.shape
    #     return zoom(data, zoom_factors, order=0).astype(np.float32)
    #
    # dataset = dataset.map(lambda data: tuple(tf.py_func(_resize_py_function,
    #                                                     [data],
    #                                                     [tf.float32])))
    # def _resize_py_function(data, target_affine, target_shape):
    #     zoom_factors = target_shape/data.shape
    #     return zoom(data, zoom_factors, order=0).astype(np.float32)
    #
    # target_shape = tf.constant(np.array([64, 64, 64]))
    # dataset = dataset.map(lambda data: tuple(tf.py_func(_resize_py_function,
    #                                                     [data, target_shape],
    #                                                     [tf.float32])))
    # sess = tf.Session()
    dataset = tf.data.Dataset.zip((dataset, dataset))
    dataset = dataset.cache(cache)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(n_epochs)
    return dataset, target_shape


from nilearn.plotting import plot_glass_brain
import io
def gen_plot(data):
    args = {"colorbar": False,
            "plot_abs": False,
            "threshold": 0,
            "cmap": "RdBu",
            }
    target_affine = np.array([[4., 0., 0., -75.],
                              [0., 4., 0., -105.],
                              [0., 0., 4., -70.],
                              [0., 0., 0., 1.]])
    nii = nb.Nifti1Image(np.squeeze(data), target_affine)
    buf = io.BytesIO()
    p = plot_glass_brain(nii, black_bg=False,
                         display_mode='lyrz',
                         **args)
    p.savefig(buf)
    p.close()
    buf.seek(0)
    val = buf.getvalue()
    return val
