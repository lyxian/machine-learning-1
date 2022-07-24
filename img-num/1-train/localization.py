from random import sample
from PIL import Image
import numpy as np
import cv2
import os

EXPAND = 4
TRAIN_DIR = f'data/png'
FILENAME = '825.png' #sample(os.listdir(TRAIN_DIR), len(os.listdir(TRAIN_DIR)))[0]
NUMBER = FILENAME.split('.')[0]
SAVE_PATH = f'train/{NUMBER}'
SAVE_IMAGE = f'{SAVE_PATH}/0.png'
KERNEL_SIZE = (4, 4)
SAVE_AS_TXT = True

if not os.path.exists(SAVE_PATH):
    # for directory in os.listdir('testing'):
    #     if directory != '_':
    #         os.rmdir(f'testing/{directory}')
    os.mkdir(SAVE_PATH)
else:
    print(f'Folder {SAVE_PATH} exists')
    # import sys
    # sys.exit()

# Process Image
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
    saveStr = []

    if 0:
        d = {
            'x1': [],
            'y1': [],
            'x2': [],
            'y2': []
        }

        for idx, box in enumerate(boundingBoxes, start=1):
            x, y, w, h = box
            w += EXPAND; h += EXPAND
            x -= EXPAND; y -= EXPAND
            if x < 0 or y < 0:
                continue
            x1 = x; y1 = y; x2= x+w; y2 = y+h
            for i in d.keys():
                d[i] += [locals()[i]]

        def getThree(items, n=5):   
            idx = [i + 1 for (x, y, i) in zip(items, items[1:], range(len(items))) if n < abs(x - y)]
            res = [items[start:end] for start, end in zip([0] + idx, idx + [len(items)])]
            # print(len(res), res)
            return [_[0] for _ in res]

        x1 = getThree(sorted(d['x1']))
        y1 = sorted(d['y1'])[0]
        x2 = sorted(getThree(sorted(d['x2'], reverse=True)))
        y2 = sorted(d['y2'], reverse=True)[0]
        print(x1,y1,x2,y2)
    else:
        for idx, box in enumerate(boundingBoxes, start=1):
            x, y, w, h = box
            w += EXPAND; h += EXPAND
            x -= EXPAND; y -= EXPAND
            a = cv2.rectangle(tmp, (x, y), (x+w, y+h), (255, 0, 0), 1)
            cv2.imwrite(f'{SAVE_PATH}/{idx}.png', img[y:y+h,x:x+w])
            print(f'Saved to {SAVE_PATH}/{idx}.png ...')
            saveStr += [f'{idx}: ({x}, {y}) -> ({x+w}, {y+h}) | w={w}, h={h}']

        if SAVE_AS_TXT:
            with open(f'{SAVE_PATH}/0.txt', 'w') as file:
                file.write('\n'.join(saveStr))