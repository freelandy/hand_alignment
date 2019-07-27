# coding:utf-8

import numpy as np

import config


def rms_error(aligned_shape,true_shape):
    pts_eval = config.pts_eval
    n_pts = len(pts_eval)

    X_align = aligned_shape[pts_eval,:]
    X_true = true_shape[pts_eval,:]

    sum = 0

    # compute rms
    for i in range(n_pts):
        sum += np.linalg.norm(X_align[i,:] - X_true[i,:])

    rms = sum/n_pts

    # normalize the error
    centroid_left = np.mean(X_true[config.keypoint_1_idx,:],axis=0)
    centroid_right = np.mean(X_true[config.keypoint_2_idx,:],axis=0)
    distance = np.linalg.norm(centroid_left - centroid_right)

    return rms/distance

