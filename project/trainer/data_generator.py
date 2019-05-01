import math
import os
import random
from random import shuffle

import cv2 as cv
import numpy as np
from keras.utils import Sequence
import tensorflow as tf

from trainer.config import batch_size
from trainer.config import fg_path, bg_path, a_path
from trainer.config import train_names_path, valid_names_path
from trainer.config import img_cols, img_rows, channel
from trainer.config import unknown_code
from trainer.config import fg_names_path, bg_names_path
from trainer.config import skip_crop
from trainer.utils import safe_crop, crop
import trainer.my_io as mio

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

fg_files = mio.read_lines(fg_names_path)
bg_files = mio.read_lines(bg_names_path)

def get_alpha(name):
    fg_i = int(name.split("_")[0])
    name = fg_files[fg_i]
    filename = os.path.join('data/mask', name)
    alpha = mio.imread(filename, 0)
    return alpha


def get_alpha_test(name):
    fg_i = int(name.split("_")[0])
    name = fg_test_files[fg_i]
    filename = os.path.join('data/mask_test', name)
    alpha = mio.imread(filename, 0)
    return alpha


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
    return im, a, fg, bg


def process(im_name, bg_name):
    im = mio.imread(fg_path + im_name, cache_dir=os.path.join('cache', 'fg'))
    a = mio.imread(a_path + im_name, 0, cache_dir=os.path.join('cache', 'a'))
    bg = mio.imread(bg_path + bg_name, cache_dir=os.path.join('cache', 'bg'))

    if im is None or a is None or bg is None:
        if im is None:
            bad = fg_path + im_name
        elif a is None:
            bad = a_path + im_name
        else:
            bad = bg_path + bg_name
        print("Bad image: %s" % bad)
        return None

    h, w = im.shape[:2]
    if skip_crop:
        if h > img_rows and w > img_cols:
            if h <= w:
                w = math.ceil(w / h * img_rows)
                h = img_rows
            else:
                h = math.ceil(h / w * img_cols)
                w = img_cols
            im = cv.resize(src=im, dsize=(w, h), interpolation=cv.INTER_CUBIC)
            a = cv.resize(src=a, dsize=(w, h), interpolation=cv.INTER_CUBIC)
        h, w = im.shape[:2]

    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)
    return composite4(im, bg, a, w, h)


def generate_trimap(alpha):
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    # fg = cv.erode(fg, kernel, iterations=np.random.randint(1, 3))
    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    unknown = cv.dilate(unknown, kernel, iterations=np.random.randint(1, 20))
    trimap = fg * 255 + (unknown - fg) * 128
    return trimap.astype(np.uint8)


# Randomly crop (image, trimap) pairs centered on pixels in the unknown regions.
def random_choice(trimap, crop_size=(320, 320)):
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


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        if usage == 'train':
            filename = train_names_path
        elif usage == 'valid':
            filename = valid_names_path
        else:
            raise ValueError(usage)

        self.names = mio.read_lines(filename)
        np.random.shuffle(self.names)

    def __len__(self):
        return int(np.ceil(len(self.names) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.names) - i))
        batch_x = np.empty((length, img_rows, img_cols, channel), dtype=np.float32)
        batch_y = np.empty((length, img_rows, img_cols, 2), dtype=np.float32)

        bad_images = []
        for i_batch in range(length):
            name = self.names[i]
            fcount = int(name.split('.')[0].split('_')[0])
            bcount = int(name.split('.')[0].split('_')[1])
            im_name = fg_files[fcount]
            bg_name = bg_files[bcount]
            processed = process(im_name, bg_name)
            if processed is None:
                bad_images.append(i_batch)
                print("Skipping bad image")
                i += 1
                continue

            image, alpha, fg, bg = processed

            trimap = generate_trimap(alpha)

            if not skip_crop:
                # crop size 320:640:480 = 1:1:1
                different_sizes = [(320, 320), (480, 480), (640, 640)]
                crop_size = random.choice(different_sizes)

                x, y = random_choice(trimap, crop_size)
                image = safe_crop(image, x, y, crop_size)
                alpha = safe_crop(alpha, x, y, crop_size)

            else:
                h, w = image.shape[:2]
                x = 0 if img_cols == w else (w - img_cols) // 2
                y = 0 if img_rows == h else (h - img_rows) // 2
                image = crop(image, x, y, (img_rows, img_cols))
                alpha = crop(alpha , x, y, (img_rows, img_cols))

            if channel == 4:
                trimap = generate_trimap(alpha)

            # Flip array left to right randomly (prob=1:1)
            if np.random.random_sample() > 0.5:
                image = np.fliplr(image)
                alpha = np.fliplr(alpha)
                if channel == 4:
                    trimap = np.fliplr(trimap)

            batch_x[i_batch, :, :, 0:3] = image / 255.
            if channel == 4:
                batch_x[i_batch, :, :, 3] = trimap / 255.

            if channel == 4:
                mask = np.equal(trimap, 128).astype(np.float32)
            else:
                mask = np.ones((img_rows, img_cols))

            batch_y[i_batch, :, :, 0] = alpha / 255.
            batch_y[i_batch, :, :, 1] = mask

            i += 1

        if bad_images:
            if len(bad_images) == length:
                print("Empty batch!")
            else:
                np.delete(batch_x, bad_images, 0)
                np.delete(batch_y, bad_images, 0)

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')

if __name__ == '__main__':
    i = 0
    for batch_x, _ in train_gen():
        for j in range(batch_x.shape[0]):
            cv.imshow('image',batch_x[j, :, :, 0:3])
            if channel == 4:
                cv.imshow('trimap',batch_x[j, :, :, 3])
            cv.waitKey(0)
            cv.destroyAllWindows()
            print(i)
            i = i + 1
