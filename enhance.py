# coding:utf-8
import numpy as np
import cv2


def enhance_image(image, filter):
    filtered_image = cv2.filter2D(np.float32(image), -1, filter) # cv2.filter2D must not use uint8 type
    filtered_image = np.round(
        ((filtered_image - np.min(filtered_image)) / (np.max(filtered_image) - np.min(filtered_image))) * 255)

    return filtered_image


def generate_filter(filter_size, sigma, mu):
    half_filter_size = np.floor(filter_size / 2)

    x, y = np.meshgrid(np.linspace(-half_filter_size, half_filter_size, filter_size),
                       np.linspace(-half_filter_size, half_filter_size, filter_size))
    r = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) * np.sin(
        2 * np.pi * mu * np.sqrt(x ** 2 + y ** 2))
    r = r - np.mean(r)

    return r
