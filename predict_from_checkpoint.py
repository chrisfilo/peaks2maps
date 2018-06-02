import numpy.linalg as npl

import nibabel as nb
import numpy as np

def get_checkpoint_dir():
    from tarfile import TarFile
    from lzma import LZMAFile
    import requests
    from io import BytesIO
    from appdirs import AppDirs
    import os
    dirs = AppDirs("peak2maps", "chrisfilo", version="1.0")
    checkpoint_dir = os.path.join(dirs.user_data_dir, "ohbm2018_model")
    print(checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        print("Downloading the model...")
        request = requests.get("https://zenodo.org/record/1257721/files/ohbm2018_model.tar.xz?download=1")
        print("Uncompressing the model...")
        tarfile = TarFile(fileobj=LZMAFile(BytesIO(request.content)), mode="r")
        tarfile.extractall(dirs.user_data_dir)
    return checkpoint_dir


def peaks2maps(contrasts_coordinates, contrasts_output_filenames):
    """

    :param contrasts_coordinates:
    :param contrasts_output_filenames:
    :return:
    """
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.DEBUG)
    import models.unet as model
    from datasets import save_nii, _get_resize_arg
    target_shape = (32, 32, 32)

    def generate_input_fn():
        def get_generator():
            affine, _ = _get_resize_arg(target_shape)
            for contrast in contrasts_coordinates:
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

    print("begin inference")
    model = tf.estimator.Estimator(model.model_fn, model_dir=get_checkpoint_dir())
    for output_filename in contrasts_output_filenames:
        results = model.predict(generate_input_fn)
        results = [result for result in results]
        assert len(results) == 1
        save_nii(results[0], target_shape, output_filename)


if __name__ == '__main__':
    peaks2maps([[[59, -44, 17]]], ["C:/scratch/test_prediction.nii.gz"])

