from .models import unet
from .datasets import _get_resize_arg

import numpy.linalg as npl
import nibabel as nb
import numpy as np
from tarfile import TarFile
from lzma import LZMAFile
import requests
from io import BytesIO
from appdirs import AppDirs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

target_shape = (32, 32, 32)


def get_generator(contrasts_coordinates, target_shape, affine, skip_out_of_bounds=False):
    def generator():
        for contrast in contrasts_coordinates:
            encoded_coords = np.zeros(list(target_shape))
            for real_pt in contrast:
                vox_pt = np.rint(nb.affines.apply_affine(npl.inv(affine), real_pt)).astype(int)
                if skip_out_of_bounds and (vox_pt[0] >= 32 or vox_pt[1] >= 32 or vox_pt[2] >= 32):
                    continue
                encoded_coords[vox_pt[0], vox_pt[1], vox_pt[2]] = 1
            yield (encoded_coords, encoded_coords)

    return generator


def get_checkpoint_dir():
    dirs = AppDirs("peak2maps", "chrisfilo", version="1.0")
    checkpoint_dir = os.path.join(dirs.user_data_dir, "ohbm2018_model")
    if not os.path.exists(checkpoint_dir):
        print("Downloading the model... (this can take a while)")
        request = requests.get("https://zenodo.org/record/1257721/files/ohbm2018_model.tar.xz?download=1")
        print("Uncompressing the model to %s..."%checkpoint_dir)
        tarfile = TarFile(fileobj=LZMAFile(BytesIO(request.content)), mode="r")
        tarfile.extractall(dirs.user_data_dir)
    return checkpoint_dir


def peaks2maps(contrasts_coordinates, skip_out_of_bounds=False, tf_verbosity_level=tf.logging.FATAL):
    """

    :param tf_verbosity_level:
    :param contrasts_coordinates:
    :param contrasts_output_filenames:
    :return:
    """
    affine, _ = _get_resize_arg(target_shape)
    tf.logging.set_verbosity(tf_verbosity_level)

    def generate_input_fn():
        dataset = tf.data.Dataset.from_generator(get_generator(contrasts_coordinates, target_shape, affine,
                                                               skip_out_of_bounds=skip_out_of_bounds),
                                                 (tf.float32, tf.float32),
                                                 (tf.TensorShape(target_shape), tf.TensorShape(target_shape)))
        dataset = dataset.batch(1)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    print("begin inference")
    model = tf.estimator.Estimator(unet.model_fn, model_dir=get_checkpoint_dir())

    results = model.predict(generate_input_fn)
    results = [result for result in results]
    assert len(results) == len(contrasts_coordinates), "returned %d" % len(results)

    niis = [nb.Nifti1Image(np.squeeze(result), affine) for result in results]
    return niis


def get_model_input(contrasts_coordinates, skip_out_of_bounds=False):
    affine, _ = _get_resize_arg(target_shape)
    niis = []
    generator = get_generator(contrasts_coordinates, target_shape, affine, skip_out_of_bounds=skip_out_of_bounds)
    for (map, _) in generator():
        nii = nb.Nifti1Image(map, affine)
        niis.append(nii)
    return niis


def parse_gingerale_txt(path):
    content = open(path).read()
    titles = []
    coordinates_list = []
    for contrast in content.split("\n\n"):
        if len(contrast) > 2:
            header_lines = [line[3:] for line in contrast.split('\n') if line.startswith("//")]
            titles.append(header_lines[-2])

            coordinates = [line for line in contrast.split('\n') if not line.startswith("//")]
            coordinates = [[float(coord) for coord in coords.split('\t')] for coords in coordinates]
            coordinates_list.append(coordinates)

    return titles, coordinates_list


if __name__ == '__main__':
    peaks2maps([[[59, -44, 17]]], ["C:/scratch/test_prediction.nii.gz"])

