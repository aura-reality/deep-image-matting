import argparse

import cv2 as cv
import numpy as np
import glob
import os


from trainer.segnet import build_encoder_decoder, build_refinement
from trainer.utils import get_final_output
from trainer.config import channel

from trainer.utils import compute_mse_loss, compute_sad_loss


# python test.py -i "images/image.png" -t "images/trimap.png"

if __name__ == '__main__':
    img_rows, img_cols = 320, 320


    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint_path", help="path to the model checkpoint")
    ap.add_argument("-i", "--test_path", help="path to the test images")
    ap.add_argument("-m", "--test_mask_path", help="path to the test masks")
    ap.add_argument("-s", "--stage", help="stage of model to build, should be the same as the checkpoint")

    args = vars(ap.parse_args())

    checkpoint_path = args["checkpoint_path"]
    test_path = args["test_path"]
    test_mask_path = args["test_mask_path"]
    stage = args["stage"]

    if stage is None:
        stage = 'encoder_decoder'


    model = build_encoder_decoder()
    if stage == 'refinement':
        model = build_refinement(model)
    model.load_weights(checkpoint_path)


    files = glob.glob(test_path + "/*.jpg")
    num_images = len(files)

    test_images = np.empty((num_images, img_rows, img_cols, channel), dtype=np.float32) 
    test_masks = np.empty((num_images,img_rows,img_cols), dtype=np.uint8)

    count = 0
    for f in os.listdir(test_path): 
        if f.endswith(".jpg"):     
            img = cv.imread(os.path.join(test_path, f))
            mask = cv.imread(os.path.join(test_mask_path, f),0)
            
            img = cv.resize(img, (img_rows, img_cols))
            mask = cv.resize(mask, (img_rows, img_cols))
            
            test_masks[count, :, :] = mask
            test_images[count, :, :, 0:3] = img / 255.
            count = count + 1

    print("Starting evaluation...")
    out = model.predict(test_images)
    out = np.reshape(out, (num_images, img_rows, img_cols))
    out = out * 255
    out = out.astype(np.uint8)

    mse_loss = []
    sad_loss = []
    trimap = np.ones((img_rows, img_cols), dtype=np.uint8)*128
    for i in range(num_images):
        mse_loss.append(compute_mse_loss(out[i], test_masks[i], trimap))
        sad_loss.append(compute_sad_loss(out[i], test_masks[i], trimap))
        
    print("SAD Loss: " + str(np.mean(sad_loss)))
    print("MSE Loss: " + str(np.mean(mse_loss)))



