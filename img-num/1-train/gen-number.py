from PIL import Image
import subprocess
import numpy
import time
import sys
import re
import os

DIR = 'data'
SUB_DIR = 'img'
FILES = [f'{DIR}/{SUB_DIR}/{_}' for _ in os.listdir(f'{DIR}/{SUB_DIR}') if '.png' in _]

# SAVE_TYPE = 'txt'
SAVE_TYPE = 'png'

IMAGE_THRESHOLD = ''
IMG_SIZE = ''
IMG_SHRINK = ''
OPTION_1 = {
    'IMAGE_THRESHOLD': 170,
    'IMG_SIZE': numpy.array([145,70]),
    'IMG_SHRINK': 0.4
}
OPTION_2 = {
    'IMAGE_THRESHOLD': 185,
    'IMG_SIZE': numpy.array([145,70]),
    'IMG_SHRINK': 0.6
}
for k, v in OPTION_1.items():
# for k, v in OPTION_2.items():
    globals()[k] = v

if len(sys.argv) == 2:
    IMG_SHRINK = float(sys.argv[1])

# Create folder
if os.path.exists(f'{DIR}/{SAVE_TYPE}/{IMG_SHRINK}'):
    subprocess.run(f'rm -r {DIR}/{SAVE_TYPE}/{IMG_SHRINK}'.split())
os.mkdir(f'{DIR}/{SAVE_TYPE}/{IMG_SHRINK}')

for file in FILES:
    img = Image.open(file).convert('L').resize(map(int,IMG_SIZE*IMG_SHRINK), Image.Resampling.LANCZOS)
    data = numpy.array(img)
    number = re.search(rf'{DIR}/{SUB_DIR}/(.*).png', file).group(1)
    savePath = f'{DIR}/{SAVE_TYPE}/{IMG_SHRINK}/{number}.{SAVE_TYPE}'

    if os.path.exists(savePath):
        os.remove(savePath)

    if SAVE_TYPE == 'txt':
        with open(savePath, 'a') as file:
            for row in data:
                file.write(' '.join(['O' if i > IMAGE_THRESHOLD else '-' for i in row])+'\n')
    elif SAVE_TYPE == 'png':
        newData = numpy.array([[0 if col > IMAGE_THRESHOLD else 255 for col in row] for row in data]).astype(numpy.uint8)
        Image.fromarray(newData).save(savePath)