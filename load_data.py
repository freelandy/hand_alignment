# coding:utf-8

import os

from PIL import Image
import numpy as np

import config
import detect


def load(train_or_test):
    if train_or_test == 'train':
        image_path = config.train_data_path
        n_samples = config.n_train_samples
    else:
        image_path = config.test_data_path
        n_samples = config.n_test_samples

    images = np.ndarray((config.canvas_size[0], config.canvas_size[1], n_samples))
    shapes = np.ndarray((config.n_points, 2, n_samples))
    for i in range(n_samples):
        # for each image
        # step1: read image
        # step2: read points
        # step3: clip image using bounding box
        # step4: recalculate points coordinates
        # step5: resize image to canvas size
        # step6: recalculate points coordinates
        image_file_name = image_path + '\\' + 'image_{:0>4d}.jpg'.format(i + 1)
        shape_file_name = image_path + '\\' + 'image_{:0>4d}.pts'.format(i + 1)

        if not os.path.exists(image_file_name) or not os.path.exists(shape_file_name):
            raise Exception('Some image or label file(s) does not exist.')

        # step1
        img = Image.open(image_file_name)

        # step2
        shape = load_shape(shape_file_name)

        # step3
        bbox = detect.detect_hand(img, shape)
        img_new = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2] - 1, bbox[1] + bbox[3] - 1))

        # step4
        shape_new = shape - np.repeat([[bbox[0], bbox[1]]], shape.shape[0], axis=0)

        # step5
        sr = config.canvas_size[0] / img_new.height
        sc = config.canvas_size[1] / img_new.width
        img_new = img_new.resize((config.canvas_size[0], config.canvas_size[1]), Image.BICUBIC)
        if img_new.mode == 'RGB':
            img_new = img_new.convert('L')  # convert to gray scale

        # step6
        shape_new = shape_new * np.repeat([[sc, sr]], shape.shape[0], axis=0)  # sc->x-coordinate, sr->y-ccordinate

        images[:, :, i] = img_new
        shapes[:, :, i] = shape_new

    return images, shapes


def load_shape(shape_file_name):
    shape = np.ndarray((config.n_points, 2), )
    with open(shape_file_name) as fp:
        line = fp.readline()

        line_cnt = 1
        num_pts = 0
        while line and num_pts < config.n_points:
            line_cnt = line_cnt + 1
            line = fp.readline()

            if line_cnt > 3:
                pt = [float(line.split()[0]), float(line.split()[1])]
                shape[num_pts, :] = np.array(pt)
                num_pts = num_pts + 1

    return shape


if __name__ == '__main__':
    images, shapes = load('train')
