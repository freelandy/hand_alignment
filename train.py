# coding:utf-8

import numpy as np
from PIL import Image
import cv2

import config
import load_data
import local_descriptors
import regression
import error_compute
from model import Model


def do_training():
    images, shapes = load_data.load('train')

    # set the mean shape as the initial shape
    mean_shape = np.mean(shapes, axis=2)
    temp_shapes = np.tile(mean_shape[:, :, np.newaxis], (1, 1, shapes.shape[2]))  # this usage is rather odd

    model = Model('model.bin')
    model.m = mean_shape
    for cascade in range(config.n_cascades):
        r, new_shapes, rms = learn_one_cascade(cascade, images, shapes, temp_shapes)
        temp_shapes = new_shapes

        model.R.append(r)

        if config.verbose: print('Training error:\t{:f}\n'.format(rms))

    model.save()


def learn_one_cascade(current_cascade, images, ground_truth_shapes, temp_shapes):
    # scale factor of current cascade
    scale = 1 / (config.scale_factor ** (config.n_cascades - (current_cascade + 1)))

    delta_shapes = []
    descs = []  # local descriptors may have various length according to different scales
    for i in range(images.shape[2]):
        if config.verbose: print('Cascade: {:d}\tSample: {:d}\n'.format(current_cascade + 1, i + 1))

        # compute the initial shape
        # for the first cascade, the mean shape is used as the initial shape
        # align the mean shape according to the bounding box of the groundtruth shape (why to do so?)
        if current_cascade == 0:
            # here to align the mean shape according to the bounding box of the groundtruth shape
            pass

        # resize image and shape according to current cascade
        img = Image.fromarray(images[:, :, i])
        img = img.resize((np.floor(img.size[0] * scale), np.floor(img.size[1] * scale)), Image.BICUBIC)
        shape = temp_shapes[:, :, i] * scale

        # extract local descriptors
        desc, desc_size = local_descriptors.hog(np.asarray(img), shape, current_cascade)
        descs.append(
            desc)  # shape of descs: n*m, n is the number of samples, m is the length of local descriptors of one image

        # compute delta shape (current shape minus true shape)
        delta_shape = shape - ground_truth_shapes[:, :, i] * scale
        delta_shape = np.true_divide(delta_shape, np.tile(
            [np.max(shape[:, 0]) - np.min(shape[:, 0]), np.max(shape[:, 1]) - np.min(shape[:, 1])],
            (shape.shape[0], 1)))  # normalize delta_shape
        delta_shapes.append(delta_shape.ravel().tolist())  # reshape along rows

    # solving multivariant linear regression
    if config.verbose: print('Solving linear regression problem...\n')

    delta_shapes = np.array(delta_shapes)
    descs = np.array(descs)

    # (X'*X+eye(descs.shape[1])*alpha)\X'*Y
    a = descs.T.dot(descs) + (np.eye(descs.shape[1]) * config.alpha[current_cascade])
    b = np.linalg.inv(a).dot(descs.T)
    R = b.dot(delta_shapes)

    # update shapes
    if config.verbose: print('Updating shapes...\n')

    temp_delta_shapes = regression.regress(descs, R)  # n_training_sample by n_points * 2

    new_shapes = np.ndarray(temp_shapes.shape)
    for i in range(images.shape[2]):
        shape = temp_shapes[:, :, i] * scale
        orig_delta_shapes = np.reshape(temp_delta_shapes[i, :], (-1, 2)) * np.tile(
            [np.max(shape[:, 0]) - np.min(shape[:, 0]), np.max(shape[:, 1]) - np.min(shape[:, 1])],
            (shape.shape[0], 1))  # de-normalize estimated delta_shape. reshape along rows
        new_shapes[:, :, i] = (shape - orig_delta_shapes) / scale

    # compute error
    err = np.zeros((images.shape[2], 1))
    for i in range(images.shape[2]):
        err[i] = error_compute.rms_error(new_shapes[:, :, i], ground_truth_shapes[:, :, i])

    rms = np.mean(err) * 100

    return R, new_shapes, rms


if __name__ == '__main__':
    do_training()
