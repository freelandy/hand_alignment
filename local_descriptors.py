# coding:utf-8

import numpy as np
import cv2  # skimage do not have sift descriptors, I have to uese cv2. My egg ache.
# from skimage import feature
# from PIL import Image

import config



def hog(im, pts, current_cascade):
    lmsize = np.int(np.floor(config.desc_scale[current_cascade] * im.shape[0]))

    desc = []
    for pt in pts:
        # crop a small image patch around pt
        left_x = np.round(pt[0] - (lmsize - 1) / 2).astype(np.int)
        top_y = np.round(pt[1] - (lmsize - 1) / 2).astype(np.int)
        # right_x = np.round(left_x + lmsize - 1).astype(np.int)
        # bottom_y = np.round(top_y + lmsize - 1).astype(np.int)
        # im_patch = im.crop((left_x, top_y, right_x, bottom_y)) # use PIL and skimage
        # by using python-opencv, im is a ndarray in numpy

        im_patch = np.zeros((lmsize, lmsize))
        for r in range(im_patch.shape[0]):
            if r + top_y > im.shape[0] - 1:
                continue
            for c in range(im_patch.shape[1]):
                if c + left_x > im.shape[1] - 1:
                    continue
                im_patch[r, c] = im[r + top_y, c + left_x]

        if im_patch.shape[0] != lmsize or im_patch.shape[1] != lmsize:
            im_patch = cv2.resize(im_patch, (int(lmsize), int(lmsize)), interpolation=cv2.INTER_CUBIC)

        # this is different with that implemented by vl_feat, maybe in sliding steps
        # it will automatically flattern feature map into feature vector

        # hog_vector = feature.hog(im_patch, orientations=9, pixels_per_cell=dsize, cells_per_block=(2, 2)) # use IPL and skimage
        winSize = (np.int(im_patch.shape[0]), np.int(im_patch.shape[1]))
        blockSize = (np.int(2 * config.desc_cell_size), np.int(2 * config.desc_cell_size))
        blockStride = (np.int(blockSize[0] / 2), np.int(blockSize[1] / 2))
        cellSize = (np.int(config.desc_cell_size), np.int(config.desc_cell_size))
        nbins = 9
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize,
                                nbins)  # all these parameters must be type of integer
        hog_vector = hog.compute(np.uint8(im_patch))
        desc.extend(hog_vector.ravel().tolist())

    desc_size = len(hog_vector)
    return desc, desc_size


def sift(im):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(np.uint8(im), None)  # sift feature extract must use uint8 type

    return kp, desc


def match_sift(desc1, desc2, n_matches):
    bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)

    # return n_matches best matched points
    return matches[:n_matches]
