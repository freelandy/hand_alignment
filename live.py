# coding:utf-8

import cv2
from PIL import Image

import alignment
import config
from model import Model

model = Model('model.bin').load()

camera = cv2.VideoCapture(0)

while camera.isOpened():
    (grabbed, frame) = camera.read()

    # crop center part
    roi = frame[:, 309:909, :]
    sr = roi.shape[0] / config.canvas_size[0]
    sc = roi.shape[1] / config.canvas_size[1]
    # resize
    roi = cv2.resize(roi, (config.canvas_size[0], config.canvas_size[1]), cv2.INTER_LINEAR)
    # gray scale
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # align
    aligned_shape = alignment.align_shape(model, roi)

    # plot aligned shape
    for i in range(config.n_points):
        aligned_shape[i, 0] = aligned_shape[i, 0] * sc + 309
        aligned_shape[i, 1] = aligned_shape[i, 1] * sr

        cv2.circle(frame, (int(model.m[i, 0] * sc + 309), int(model.m[i, 1] * sr)), 6, (255, 0, 0))
        cv2.circle(frame, (int(aligned_shape[i, 0]), int(aligned_shape[i, 1])), 6, (0, 255, 0))
        # cv2.circle(roi, (int(model.m[i, 0]), int(model.m[i, 1])), 6, (255, 0, 0))
        # cv2.circle(roi, (int(aligned_shape[i, 0]), int(aligned_shape[i, 1])), 6, (0, 255, 0))

    mirror = frame[:,::-1]

    cv2.imshow("Security Feed", mirror)
    key = cv2.waitKey(1)

    # 如果q键被按下，跳出循环
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
