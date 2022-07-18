from random import sample
from PIL import Image
import numpy as np
import cv2
import os

ZOOM = 1
TRAIN_DIR = f'data/png/{ZOOM}'
FILENAME = sample(os.listdir(TRAIN_DIR), len(os.listdir(TRAIN_DIR)))[0]
NUMBER = FILENAME.split('.')[0]
SAVE_PATH = f'testing/{NUMBER}'

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
else:
    print(f'Folder {SAVE_PATH} exists')
    import sys
    sys.exit()

# Process Image
SAVE_IMAGE = f'{SAVE_PATH}/0.png'
KERNEL_SIZE = (4, 4)
if 1:
    img = cv2.imread(f'{TRAIN_DIR}/{FILENAME}')
    tmp = img.copy()
    kernel = np.ones(KERNEL_SIZE, np.uint8)
    erosion = cv2.erode(tmp, kernel, iterations = 1)
    Image.fromarray(erosion).save(SAVE_IMAGE)

# Localization
if 1:
    mser = cv2.MSER_create(delta=8, min_diversity=0.1)

    img = cv2.imread(SAVE_IMAGE)
    tmp = img.copy()
    regions, boundingBoxes = mser.detectRegions(tmp)

    for idx, box in enumerate(boundingBoxes, start=1):
        x, y, w, h = box
        a = cv2.rectangle(tmp, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.imwrite(f'{SAVE_PATH}/{idx}.png', img[y:y+h,x:x+w])
        print(f'Saved to {SAVE_PATH}/{idx}.png ...')