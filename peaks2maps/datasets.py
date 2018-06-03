import matplotlib as mpl
from glob import glob
import tensorflow as tf
import os
import nibabel as nb
import numpy as np
from nilearn import image
from nilearn.plotting import plot_glass_brain
from skimage.feature import peak_local_max
import random
import io
import scipy.ndimage

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


def _get_data(nthreads, batch_size, src_folder, n_epochs, cache, shuffle,
              target_shape, add_nagatives, smoothness_levels, cluster_forming_thrs):
    paths_ds = tf.data.Dataset.list_files(os.path.join(src_folder, "*.nii.gz"),
                                           shuffle=shuffle)

    def read_and_augument(path, target_shape):

        paths_ds = tf.data.Dataset.from_tensors((path,))

        target_affine, target_shape = _get_resize_arg(target_shape)

        def _read_and_resample(path, target_affine, target_shape):
            path_str = path.decode('utf-8')
            filename = path_str.split(os.sep)[-1]
            nii = nb.load(path_str)
            data = nii.get_data()
            data[np.isnan(data)] = 0
            m = np.max(np.abs(data))
            data = data / m
            nii = nb.Nifti1Image(data, nii.affine)
            nii = image.resample_img(nii,
                                     target_affine=target_affine,
                                     target_shape=target_shape)
            data = nii.get_data().astype(np.float32)
            return data, filename

        data_ds = paths_ds.map(
            lambda path: tuple(tf.py_func(_read_and_resample,
                                          [path, target_affine,
                                           target_shape],
                                          [tf.float32, tf.string],
                                          name="read_and_resample"),
                               ),
            num_parallel_calls=2)

        def _smooth(data, filename, target_affine, smoothness_level):
            nii = nb.Nifti1Image(data, target_affine)
            nii = image.smooth_img(nii, smoothness_level)  # !!!!!!
            data = nii.get_data()
            m = np.max(np.abs(data))
            data = data / m
            data = data.astype(np.float32)
            return data, filename

        smoothed_ds = None
        for smoothness_level in smoothness_levels:
            if smoothness_level:
                tmp_smooth_ds = data_ds.map(
                    lambda data, filename: tuple(tf.py_func(_smooth,
                                                            [data, filename,
                                                             target_affine,
                                                             tf.constant(smoothness_level)],
                                                            [np.float32, tf.string],
                                                            name="smooth")),
                    num_parallel_calls=2)
            else:
                tmp_smooth_ds = data_ds

            if smoothed_ds is None:
                smoothed_ds = tmp_smooth_ds
            else:
                print("concatenate")
                smoothed_ds = smoothed_ds.concatenate(tmp_smooth_ds)

        def _resize_with_filename(data, filename):
            reshaped = tf.reshape(data, target_shape)
            return reshaped, filename

        resized_ds = smoothed_ds.map(lambda data, filename: _resize_with_filename(data, filename),
                                     num_parallel_calls=2)

        if add_nagatives:
            print("adding negatives")
            negatives = resized_ds.map(lambda data, filename: (tf.scalar_mul(-1, data), filename),
                                       num_parallel_calls=2)
            resized_ds = resized_ds.concatenate(negatives)

        def _extract_peaks(data, cluster_forming_thr):
            new = np.zeros_like(data)
            new[data > cluster_forming_thr] = 1
            labels, n_features = scipy.ndimage.label(new)
            for j in range(1, n_features+1):
                if (labels == j).sum() < 5:
                    labels[labels == j] = 0
            peaks = peak_local_max(data, indices=False, min_distance=1,
                                   num_peaks_per_label=45,
                                   labels=labels,
                                   threshold_abs=cluster_forming_thr).astype(np.float32)
            peaks[peaks > 0] = 1.0
            return peaks

        peaks_ds = None
        for cluster_forming_thr in cluster_forming_thrs:

            tmp_peaks_ds = resized_ds.map(
                lambda data, _: tuple(tf.py_func(_extract_peaks,
                                              [data,
                                               tf.constant(cluster_forming_thr)],
                                              [tf.float32])),
                num_parallel_calls=2)
            if peaks_ds is None:
                peaks_ds = tmp_peaks_ds
                resized_ds = resized_ds
                paths_ds = paths_ds
            else:
                peaks_ds = peaks_ds.concatenate(tmp_peaks_ds)
                resized_ds = resized_ds.concatenate(resized_ds)
                paths_ds = paths_ds.concatenate(paths_ds)

        def _resize(data):
            reshaped = tf.reshape(data, target_shape)
            return reshaped

        peaks_ds = peaks_ds.map(_resize)

        dataset = tf.data.Dataset.zip((peaks_ds, resized_ds))

        def _filter_empty(peaks, _):
            return tf.reduce_sum(peaks) > 0
        return dataset.filter(_filter_empty)

    dataset = paths_ds.apply(
        tf.contrib.data.parallel_interleave(
            lambda path: read_and_augument(path, target_shape),
            cycle_length=4))
    if cache:
        dataset = dataset.cache(cache)

    if shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000, n_epochs))
    else:
        dataset = dataset.prefetch(1000)
        dataset = dataset.repeat(n_epochs)

    dataset = dataset.batch(batch_size)

    return dataset, target_shape


class Peaks2MapsDataset:

    def __init__(self, target_shape, train_batch_size,
                 validation_batch_size, n_epochs, nthreads=None):
        self.target_shape = target_shape
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.n_epochs = n_epochs
        if nthreads is None:
            import multiprocessing
            self.nthreads = multiprocessing.cpu_count()
        else:
            self.nthreads = nthreads

    def train_input_fn(self):
        self.training_dataset, self.target_shape = _get_data(8,
                                                             self.train_batch_size,
                                                             "D:/data/neurovault/neurovault/vetted/train",
                                                             self.n_epochs,
                                                             'D:/drive/workspace/peaks2maps/cache_train_new_thr',
                                                             True,
                                                             self.target_shape,
                                                             True,
                                                             smoothness_levels=[0],
                                                             cluster_forming_thrs=[0.6])

        # You can use feedable iterators with a variety of different kinds of iterator
        # (such as one-shot and initializable iterators).
        self.training_iterator = self.training_dataset.make_one_shot_iterator()
        return self.training_iterator.get_next()

    def eval_input_fn(self):
        self.validation_dataset, validation_shape = _get_data(8,
                                                              self.validation_batch_size,
                                                              "G:/My Drive/data/neurovault/neurovault/vetted/eval", #!!!
                                                              #"D:/data/hcp_statmaps/val_all_tasks",
                                                              1,
                                                              False,
                                                              False,
                                                              self.target_shape,
                                                              False,
                                                              smoothness_levels=[8],
                                                              cluster_forming_thrs=[0.6])

        self.validation_iterator = self.validation_dataset.make_one_shot_iterator()
        return self.validation_iterator.get_next()


def get_plot_op(image, target_shape, summary_label):
    mpl.use('agg')
    target_affine, _ = _get_resize_arg(target_shape)
    target_affine_tf = tf.constant(target_affine)

    def _gen_plot(data, target_affine):
        if len(data.shape) == 4:
            data = data[0, :, :, :]
        args = {"colorbar": True,
                "plot_abs": False,
                "threshold": 0,
                "cmap": "RdBu_r",
                }
        nii = nb.Nifti1Image(np.squeeze(data), target_affine)
        buf = io.BytesIO()
        p = plot_glass_brain(nii, black_bg=False,
                             display_mode='lyrz',
                             vmin=-1,
                             vmax=1,
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
                                     image, max_outputs=10)
    return summary_image


def save_nii(data, target_shape, path):
    target_affine, _ = _get_resize_arg(target_shape)

    nii = nb.Nifti1Image(np.squeeze(data), target_affine)
    nii.to_filename(path)