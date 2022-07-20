from random import sample
from PIL import Image
import numpy as np
import cv2
import os

ZOOM = 1
EXPAND = 4
TRAIN_DIR = f'data/png/{ZOOM}'
FILENAME = sample(os.listdir(TRAIN_DIR), len(os.listdir(TRAIN_DIR)))[0]
NUMBER = FILENAME.split('.')[0]
SAVE_PATH = f'testing/{NUMBER}'
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
    import sys
    sys.exit()

# Process Image
if 1:
    img = cv2.imread(f'{TRAIN_DIR}/{FILENAME}')
    tmp = img.copy()
    kernel = np.ones(KERNEL_SIZE, np.uint8)
    erosion = cv2.erode(tmp, kernel, iterations = 1)
    Image.fromarray(erosion).save(SAVE_IMAGE)

# Localization
print(NUMBER)
if 1:
    mser = cv2.MSER_create(delta=8, min_diversity=0.1)

    img = cv2.imread(SAVE_IMAGE)
    tmp = img.copy()
    regions, boundingBoxes = mser.detectRegions(tmp)
    saveStr = []
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
        x1 = x; y1 = y; x2= x+w; y2 = y+h
        for i in d.keys():
            d[i] += [locals()[i]]

    def getThree(items, n=5):   
        idx = [i + 1 for (x, y, i) in zip(items, items[1:], range(len(items))) if n < abs(x - y)]
        res = [items[start:end] for start, end in zip([0] + idx, idx + [len(items)])]
        # print(len(res), res)
        return [_[0] for _ in res]

    # Maximize Capture
    # - Sort by x 
    #   > Get lowest + Keep 3
    # - Sort by y
    #   > Get lowest + Keep 1
    x1 = getThree(sorted(d['x1']))
    y1 = sorted(d['y1'])[0]

    
    # - Sort by x+w
    #   > Get highest + Keep 3
    # - Sort by y+h
    #   > Get highest + Keep 1
    x2 = sorted(getThree(sorted(d['x2'], reverse=True)))
    y2 = sorted(d['y2'], reverse=True)[0]

    btmLeft = list(zip(x1, [y1]*3))
    topRight = list(zip(x2, [y2]*3))

    for idx, (x1y1, x2y2) in enumerate(zip(btmLeft, topRight), start=1):
        x1, y1 = x1y1; x2, y2 = x2y2
        digit = NUMBER[idx-1]
        cv2.imwrite(f'{SAVE_PATH}/{idx}-{digit}.png', img[y1:y2,x1:x2])
        print(f'Saved to {SAVE_PATH}/{idx}-{digit}.png ...')

if 1:
    os.remove(SAVE_IMAGE)
    # os.rmdir(SAVE_PATH)
    print(f'{SAVE_PATH} removed..')