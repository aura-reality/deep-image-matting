import math
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

import sys

from config import img_rows, img_cols, unknown_code
from model import build_encoder_decoder, build_refinement
from utils import get_final_output, safe_crop, draw_str

def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, bg

# Randomly crop (image, trimap) pairs centered on pixels in the unknown regions.
def random_choice(trimap, crop_size):
    crop_height, crop_width = crop_size
    y_indices, x_indices = np.where(trimap == unknown_code)
    num_unknowns = len(y_indices)
    x, y = 0, 0
    if num_unknowns > 0:
        ix = np.random.choice(range(num_unknowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]
        x = max(0, center_x - int(crop_width / 2))
        y = max(0, center_y - int(crop_height / 2))
    return x, y

def load_model():
    pretrained_path = 'models/final.42-0.0398.hdf5'
    encoder_decoder = build_encoder_decoder()
    final = build_refinement(encoder_decoder)
    final.load_weights(pretrained_path)
    print(final.summary())
    return final

def matte(image_path, trimap_path, model): 

    # Read the background image
    #
    bgr_img = cv.imread(image_path)
    bg_h, bg_w = bgr_img.shape[:2]
    print('bg_h, bg_w: ' + str((bg_h, bg_w)))

    # Read the trimap in grayscale
    trimap = cv.imread(trimap_path, 0)

    # Crop
    different_sizes = [(320, 320), (320, 320), (320, 320), (480, 480), (640, 640)]
    crop_size = random.choice(different_sizes)
    x, y = random_choice(trimap, crop_size)
    print('x, y: ' + str((x, y)))

    bgr_img = safe_crop(bgr_img, x, y, crop_size)
    trimap = safe_crop(trimap, x, y, crop_size)
    cv.imwrite('matting/image.png', np.array(bgr_img).astype(np.uint8))
    cv.imwrite('matting/trimap.png', np.array(trimap).astype(np.uint8))

    x_test = np.empty((1, img_rows, img_cols, 4), dtype=np.float32)
    x_test[0, :, :, 0:3] = bgr_img / 255.
    x_test[0, :, :, 3] = trimap / 255.

    y_pred = model.predict(x_test)
    # print('y_pred.shape: ' + str(y_pred.shape))

    y_pred = np.reshape(y_pred, (img_rows, img_cols))
    print(y_pred.shape)
    y_pred = y_pred * 255.0
    y_pred = get_final_output(y_pred, trimap)
    y_pred = y_pred.astype(np.uint8)

    out = y_pred.copy()
    cv.imwrite('matting/out.png', out)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: matte.py image.png trimap.png")
        sys.exit(1)

    image_path = sys.argv[1]   
    trimap_path = sys.argv[2]

    model = load_model()
    matte(image_path, trimap_path, model)
    K.clear_session()
