import tensorflow as tf

from datasets import Peaks2MapsDataset, _get_data, save_nii
import models.unet as model
import models.fixed_conv as reference_model
import os, datetime


with tf.Session() as sess:
    # validation_dataset, validation_shape = _get_data(8, 1, #
    #                                                  "D:/data/hcp_statmaps/test",
    #                                                  1,
    #                                                  'bogus',
    #                                                  False,
    #                                                  (32, 32, 32),
    #                                                  True,
    #                                                  smoothness_levels = [0, 10],
    #                                                  cluster_forming_thrs = [0.65])
    validation_dataset, validation_shape = _get_data(8,
              1,
              "D:/data/neurovault/neurovault/vetted/train",
              1,
              'D:/drive/workspace/peaks2maps/cache_train',
              True,
              (32, 32, 32),
              False,
              smoothness_levels=[4],  # 6, 8],
              cluster_forming_thrs=[0.6])

    validation_iterator = validation_dataset.make_one_shot_iterator()
    filenames = []
    count = 0

    import time

    start = time.time()
    print("hello")

    while True:
        try:
            # out = sess.run(
            #     validation_iterator.get_next())
            # print(out)
            features, (labels, filename) = sess.run(validation_iterator.get_next())
            # filename = filename[0][0].decode('utf-8')
            # save_nii(features, (32, 32, 32), "D:/data/hcp_statmaps/test/preprocessed/f_"  + str(count) + "_" + filename)
            # save_nii(labels, (32, 32, 32),
            #          "D:/data/hcp_statmaps/test/preprocessed/l_" + str(count) + "_" + filename)
            print(filename, count)
            # filenames.append(filename)
            count += 1
        except tf.errors.OutOfRangeError:
            break
    end = time.time()
    print(end - start)
    print(count)