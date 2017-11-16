import matplotlib as mpl
mpl.use('agg')
from glob import glob
import tensorflow as tf
import os
import nibabel as nb
import numpy as np
from scipy.ndimage.interpolation import zoom
from nilearn import image
from nilearn.plotting import plot_glass_brain
import io


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


def _get_data(nthreads, batch_size, src_folder, n_epochs, cache, shuffle,
              target_resolution):
    filenames = tf.constant(glob(os.path.join(src_folder, "*.nii.gz")))
    dataset = tf.data.Dataset.from_tensor_slices((filenames,))

    target_affine, target_shape = _get_resize_arg(target_resolution)

    def _read_py_function(filename, target_affine, target_shape):
        nii = image.resample_img(
            filename.decode('utf-8'),
            target_affine=target_affine, target_shape=target_shape)
        data = nii.get_data()
        data = data.astype(np.float32)
        return data

    target_affine_tf = tf.constant(target_affine)
    target_shape_tf = tf.constant(target_shape)
    dataset = dataset.map(
        lambda filename: tuple(tf.py_func(_read_py_function,
                                          [filename,
                                           target_affine_tf,
                                           target_shape_tf],
                                          [tf.float32])),
        num_parallel_calls=nthreads)

    def _resize(data):
        return tf.reshape(data, target_shape)

    dataset = dataset.map(_resize)

    dataset = tf.data.Dataset.zip((dataset, dataset))
    dataset = dataset.cache(cache)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(n_epochs)
    return dataset, target_shape

class Peaks2MapsDataset:

    def __init__(self, target_resolution, train_batch_size,
                 test_batch_size, n_epochs, nthreads=None):
        self.target_resolution = target_resolution
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.n_epochs = n_epochs
        if nthreads is None:
            import multiprocessing
            self.nthreads = multiprocessing.cpu_count()
        else:
            self.nthreads = nthreads

        self.training_dataset, self.target_shape = _get_data(self.nthreads,
                                                                  self.train_batch_size,
                                                                     "D:/data/hcp_statmaps/train",
                                                                     100,
                                                                     'D:/drive/workspace/peaks2maps/cache_train',
                                                                     True,
                                                                  self.target_resolution)
        self.validation_dataset, validation_shape = _get_data(self.nthreads,
                                                              self.test_batch_size,
                                                              "D:/data/hcp_statmaps/val",
                                                              1,
                                                              'D:/drive/workspace/peaks2maps/cache_val',
                                                              False,
                                                              self.target_resolution)
        assert(self.target_shape == validation_shape)

        self.handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            self.handle, self.training_dataset.output_types,
            self.training_dataset.output_shapes)
        self.input_image, self.target_image = iterator.get_next()

        # You can use feedable iterators with a variety of different kinds of iterator
        # (such as one-shot and initializable iterators).
        self.training_iterator = self.training_dataset.make_one_shot_iterator()
        self.validation_iterator = self.validation_dataset.make_initializable_iterator()


    def get_plot_op(self, image, summary_label):
        target_affine, _ = _get_resize_arg(self.target_resolution)
        target_affine_tf = tf.constant(target_affine)

        def _gen_plot(data, target_affine):
            args = {"colorbar": False,
                    "plot_abs": False,
                    "threshold": 0,
                    "cmap": "RdBu",
                    }
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

        plot = tf.py_func(_gen_plot, [image, target_affine_tf], tf.string)
        image = tf.image.decode_png(plot)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        summary_image = tf.summary.image(summary_label,
                                         image, max_outputs=200000)
        return summary_image

