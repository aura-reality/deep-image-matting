import argparse

import cv2 as cv
import numpy as np
from trainer.config import img_rows, img_cols

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-x0")
    ap.add_argument("-y0")
    ap.add_argument("-x1")
    ap.add_argument("-y1")
    args = vars(ap.parse_args())
    x0 = int(args["x0"])
    x1 = int(args["x1"])
    y0 = int(args["y0"])
    y1 = int(args["y1"])

    trimap = np.zeros((img_rows, img_cols, 1), dtype=np.uint8)
    trimap[ x0:x1, y0:y1, 0] = 128

    cv.imshow('trimap', trimap)
    cv.imwrite('made-trimap.png', trimap)
