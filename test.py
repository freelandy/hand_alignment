# coding:utf-8

from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

import config
import load_data
import error_compute
import alignment
from model import Model


def do_testing():
    images, shapes = load_data.load('test')
    model = Model('model.bin').load()

    err = np.zeros((images.shape[2],1))
    start = datetime.now()
    for i in range(images.shape[2]):
        if(config.verbose): print('Image No. \t{:d}\n'.format(i+1))

        img = Image.fromarray(images[:,:,i])
        shape = shapes[:,:,i]

        aligned_shape = alignment.align_shape(model,np.asarray(img))

        # for debug
        # draw_point(img,model.m,'green')
        # img = Image.fromarray(images[:, :, i])
        # draw_point(img,aligned_shape,'red')


        # compute error
        err[i] = error_compute.rms_error(aligned_shape,shape)

    end = datetime.now()
    print('Elapsed time for every singel sample: \t{:f} s\n'.format((end-start).microseconds/1000000/config.n_test_samples))
    print('Average error: \t{:f}\n'.format(np.mean(err)*100))



def draw_point(img,pts,color):
    draw = ImageDraw.Draw(img)

    for i in range(pts.shape[0]):
        draw.ellipse((pts[i,0]-2,pts[i,1]-2,pts[i,0]+2,pts[i,1]+2),color)

    del draw

    img.show()



if __name__=='__main__':
    do_testing()