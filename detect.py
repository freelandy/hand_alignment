# coding:utf-8

import numpy as np


def detect_hand(img, shape):
    bbox = np.array([0] * 4)
    [left_x, top_y] = np.min(shape, axis=0)
    [right_x, bottom_y] = np.max(shape, axis=0)

    # bbox[0], bbox[1]: top left corner; bbox[2], bbox[3]: width and height
    bbox[0] = left_x
    bbox[1] = top_y
    bbox[2] = right_x - left_x + 1
    bbox[3] = bottom_y - top_y + 1

    # enlarge this bbox
    scale = 2.0
    bbox[0] = np.floor(bbox[0] - (scale - 1) / 2 * bbox[2])
    bbox[1] = np.floor(bbox[1] - (scale - 1) / 2 * bbox[3])
    bbox[2] = np.floor(scale * bbox[2])
    bbox[3] = np.floor(scale * bbox[3])

    # to ensure that boudning box be in image region
    bbox[0] = np.max([bbox[0], 0])
    bbox[1] = np.max([bbox[1], 0])

    right_x = np.min([bbox[0] + bbox[2] - 1, img.width - 1])
    bottom_y = np.min([bbox[1] + bbox[3] - 1, img.height - 1])
    bbox[2] = right_x - bbox[0] + 1
    bbox[3] = bottom_y - bbox[1] + 1

    return bbox
