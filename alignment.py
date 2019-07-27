# coding:utf-8

# from PIL import Image
import numpy as np
import cv2  # skimage do not have sift descriptors, I have to uese cv2. My egg ache.

import local_descriptors
import config
import regression
import enhance


def align_shape(model, img):
    init_shape = model.m
    r = model.R

    for cascade in range(config.n_cascades):
        scale = 1 / (config.scale_factor ** (config.n_cascades - (cascade + 1)))

        img = cv2.resize(img,(np.floor(img.shape[1] * scale).astype(np.int), np.floor(img.shape[0] * scale).astype(np.int)), interpolation=cv2.INTER_CUBIC)
        init_shape = init_shape * scale

        # extract local features
        desc, desc_size = local_descriptors.hog(img, init_shape, cascade)

        # regressing
        delta_shape = regression.regress(np.array(desc), np.array(r[cascade]))

        # estimate new shape
        orig_delta_shape = np.reshape(delta_shape, (-1, 2)) * np.tile(
            [np.max(init_shape[:, 0]) - np.min(init_shape[:, 0]), np.max(init_shape[:, 1]) - np.min(init_shape[:, 1])],
            (init_shape.shape[0], 1))  # de-normalize estimated delta_shape. reshape along rows
        aligned_shape = (init_shape - orig_delta_shape) / scale

        init_shape = aligned_shape

    return aligned_shape


def align_image(image1, image2):
    # align image2 to image1, if failed, return None
    # step0: enhance images
    # step1: extract SIFT feature from both image1 and image2
    # step2: apply RANSAC to key points, find inliers
    # step3: align image to image1 using inliers

    # step0
    ehanced_image1 = enhance.enhance_image(image1, config.f)
    ehanced_image2 = enhance.enhance_image(image2, config.f)

    # step1
    kp1, desc1 = local_descriptors.sift(ehanced_image1)
    kp2, desc2 = local_descriptors.sift(ehanced_image2)
    matches = local_descriptors.match_sift(desc1, desc2, 10)  # get 10 best matched points

    # step2, findHomography only receive float points
    pts_src = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,
                                                                        2)  # src should  be image2, because we want to align image2 to image1
    pts_dst = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)  # dst should be image1
    h, mask = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC,
                                 ransacReprojThreshold=3.0)  # smaller this threshold is, more strict the RANSAC is, its typical value is 0.5

    matches = (np.array(matches)[mask.ravel() == 1]).tolist()  # remove outlier points from matches

    if len(matches) >= 3:  # could not estimate a transformation
        inliers1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        inliers2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        # step3
        transform_mat = cv2.estimateRigidTransform(inliers2, inliers1,
                                                   fullAffine=True)  # similarity transform, with affine
        if not np.array([transform_mat is None]).any():
            wrapped_image2 = cv2.warpAffine(image2, transform_mat, (image2.shape[1], image2.shape[0]))
        else:
            wrapped_image2 = None
    else:
        wrapped_image2 = None

    return wrapped_image2
