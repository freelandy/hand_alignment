# coding:utf-8
import os

import numpy as np
from PIL import Image
from datetime import datetime

import config
import load_data


def clip_roi(image, shape):
    key1 = np.mean(shape[config.keypoint_1_idx, :], axis=0)
    key2 = np.mean(shape[config.keypoint_2_idx, :], axis=0)

    w = np.linalg.norm(key2 - key1)
    h = w / 3.5

    rx = w / config.roi_size[0]
    ry = w / config.roi_size[1]

    theta = np.arctan2(key2[1] - key1[1], key2[0] - key1[0])
    if key2[1] >= key1[1]: # include horizental
        theta = theta
        theta2 = -(np.pi - theta)

        offset_x = h * np.sin(np.pi - theta)
        offset_y = h * np.cos(np.pi - theta)

    if key2[1] < key1[1]:
        theta = np.pi + theta
        theta2 = theta

        offset_x = - h * np.sin(theta)
        offset_y = h * np.cos(theta)

    roi = np.zeros(config.roi_size)
    for x in range(roi.shape[1]):
        for y in range(roi.shape[0]):
            # scale
            xx = x * rx
            yy = y * ry
            # rotate
            xxx = xx * np.cos(theta2) - yy * np.sin(theta2)
            yyy = xx * np.sin(theta2) + yy * np.cos(theta2)
            # translate
            xxx = xxx + key2[0] + offset_x
            yyy = yyy + key2[1] + offset_y

            xxx = int(np.round(xxx))
            yyy = int(np.round(yyy))

            if xxx >= 0 and xxx < image.shape[1] and yyy >= 0 and yyy < image.shape[0]:
                roi[y, x] = image[yyy, xxx]

    return roi


if __name__ == '__main__':
    images, shapes = load_data.load('test')

    # a = Image.fromarray(image)
    # if a.mode != 'RGB':
    #     a = a.convert('RGB')
    # a.save('a.jpg')

    start = datetime.now()
    for i in range(images.shape[2]):
        image = images[:, :, i]
        shape = shapes[:, :, i]
        roi = clip_roi(image, shape)

    end = datetime.now()

    print('average time:\t{:f} s'.format((end - start).microseconds / 1000000 / images.shape[2]))
