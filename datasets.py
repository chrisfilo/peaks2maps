import matplotlib as mpl
mpl.use('agg')

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
    in_images = glob(os.path.join(src_folder, "*.nii.gz"))[:300]
    if shuffle:
        random.shuffle(in_images)
    paths = tf.constant(in_images)
    paths_ds = tf.data.Dataset.from_tensor_slices((paths,))

    target_affine, target_shape = _get_resize_arg(target_shape)

    target_affine_tf = tf.constant(target_affine)
    target_shape_tf = tf.constant(target_shape)

    def _read_and_resample(path, target_affine, target_shape):
        path_str = path.decode('utf-8').split(os.sep)[-1]
        nii = nb.load(path_str)
        data = nii.get_data()
        data[np.isnan(data)] = 0
        nii = nb.Nifti1Image(data, nii.affine)
        nii = image.resample_img(nii,
                                 target_affine=target_affine,
                                 target_shape=target_shape)
        data = nii.get_data().astype(np.float32)
        tf.reshape(data, target_shape)
        return (tf.reshape(data, target_shape), path_str.split(os.sep)[-1])

    data_ds = paths_ds.map(
        lambda path: tuple(tf.py_func(_read_and_resample,
                                      [path, target_affine_tf,
                                       target_shape_tf],
                                      [tf.float32, tf.string])),
        num_parallel_calls=8)

    def smooth(data, filename):
        data = tf.data.Dataset.from_tensors(data)

        def _smooth(data, target_affine, smoothness_level):
            nii = nb.Nifti1Image(data, target_affine)
            nii = image.smooth_img(nii, smoothness_level)  # !!!!!!
            data = nii.get_data()
            m = np.max(np.abs(data))
            data = data / m
            data = data.astype(np.float32)
            return data

        smoothed_ds = None
        new_paths_ds = None
        for smoothness_level in smoothness_levels:
            sm = tf.py_func(_smooth, [data, target_affine_tf,
                                      tf.constant(smoothness_level)],
                            [np.float32])
            tmp_smooth_ds = tf.data.Dataset.from_tensors(sm)

            if smoothed_ds is None:
                smoothed_ds = tmp_smooth_ds
                new_paths_ds = paths_ds
            else:
                print("concatenate")
                smoothed_ds = smoothed_ds.concatenate(tmp_smooth_ds)
                new_paths_ds = new_paths_ds.concatenate(paths_ds)

        def _resize(data):
            return tf.reshape(data, target_shape)

        resized_ds = smoothed_ds.map(_resize, num_parallel_calls=8)

        if add_nagatives:
            print("adding negatives")
            negatives = resized_ds.map(lambda data: tf.scalar_mul(-1, data),
                                        num_parallel_calls=8)
            resized_ds = resized_ds.concatenate(negatives)
            filenames_ds = filenames_ds.concatenate(filenames_ds)

        def _extract_peaks(data, cluster_forming_thr):
            new = np.zeros_like(data)
            new[data > cluster_forming_thr] = 1
            labels, n_features = scipy.ndimage.label(new)
            for j in range(1, n_features+1):
                if (labels == j).sum() < 5:
                    labels[labels == j] = 0
            peaks = peak_local_max(data, indices=False, min_distance=0,
                                   num_peaks_per_label=1,
                                   labels=labels,
                                   threshold_abs=cluster_forming_thr).astype(np.float32)
            peaks[peaks > 0] = 1.0
            return peaks

        peaks_ds = None
        for cluster_forming_thr in cluster_forming_thrs:

            tmp_peaks_ds = resized_ds.map(
                lambda data: tuple(tf.py_func(_extract_peaks,
                                              [data,
                                               tf.constant(cluster_forming_thr)],
                                              [tf.float32])),
                num_parallel_calls=1)
            if peaks_ds is None:
                peaks_ds = tmp_peaks_ds
                resized_ds = resized_ds
                filenames_ds = filenames_ds
            else:
                peaks_ds = peaks_ds.concatenate(tmp_peaks_ds)
                resized_ds = resized_ds.concatenate(resized_ds)
                filenames_ds = filenames_ds.concatenate(filenames_ds)

        peaks_ds = peaks_ds.map(_resize)

        dataset = tf.data.Dataset.zip((peaks_ds,
                                       tf.data.Dataset.zip((resized_ds,
                                                            filenames_ds))))
        def _filter_empty(peaks, maps):
            return tf.reduce_sum(peaks) > 0
        return dataset.filter(_filter_empty)

    dataset = paths_ds.interleave(read_and_augument, cycle_length=8,
                                  block_length=16)
    dataset = dataset.prefetch(10000)
    dataset = dataset.cache()

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(n_epochs)
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
                                                             'D:/drive/workspace/peaks2maps/cache_train',
                                                             True,
                                                             self.target_shape,
                                                             False,
                                                             smoothness_levels=[4],# 6, 8],
                                                             cluster_forming_thrs=[0.6])#, 0.65, 0.7])

        # You can use feedable iterators with a variety of different kinds of iterator
        # (such as one-shot and initializable iterators).
        self.training_iterator = self.training_dataset.make_one_shot_iterator()
        return self.training_iterator.get_next()

    def eval_input_fn(self):
        self.validation_dataset, validation_shape = _get_data(8,
                                                              self.validation_batch_size,
                                                              "D:/data/neurovault/neurovault/vetted/eval", #!!!
                                                              #"D:/data/hcp_statmaps/val_all_tasks",
                                                              1,
                                                              'D:/data/hcp_statmaps/val_no_language_unseen_partitipants',
                                                              False,
                                                              self.target_shape,
                                                              False,
                                                              smoothness_levels=[6],
                                                              cluster_forming_thrs=[0.65])

        self.validation_iterator = self.validation_dataset.make_one_shot_iterator()
        return self.validation_iterator.get_next()


def get_plot_op(image, target_shape, summary_label):
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