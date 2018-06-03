from scipy.misc import imsave
import os
import nibabel as nb
import numpy as np
from glob import glob



input_dir = "D:\data\ds000030_tf2"

output_dir = "D:\data\ds000030_tf"


def process_s(t1w_file):
    print("processing %s" % t1w_file)
    mask_file = t1w_file.replace("_preproc.nii.gz", "_brainmask.nii.gz")
    bold_files = sorted(glob(
        t1w_file.replace("_T1w_preproc.nii.gz", "*space-T1w_preproc.nii.gz")))
    bold_niis = [nb.load(bold_file) for bold_file in bold_files]
    mask_nii = nb.load(mask_file)
    t1w_nii = nb.load(t1w_file)
    for i in range(mask_nii.shape[0]):
        if mask_nii.dataobj[i, :, :].sum() > 0:
            # imsave(mask_file.replace(".nii.gz", "_%03d.png") % i, np.flipud(mask_nii.dataobj[i, :, :].T))
            imsave(t1w_file.replace(".nii.gz", "_%03d.png") % i,
                   np.flipud(t1w_nii.dataobj[i, :, :].T))
            imsave(bold_files[0].replace(".nii.gz", "_%03d.png") % i,
                   np.flipud(bold_niis[0].dataobj[i, :, :].T))


from joblib import Parallel, delayed

if __name__ == '__main__':
    Parallel(n_jobs=8)(delayed(process_s)(s) for s in glob(os.path.join(input_dir, "*_T1w_preproc.nii.gz")))

