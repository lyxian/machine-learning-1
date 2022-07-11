from PIL import Image
import subprocess
import numpy
import time
import sys
import re
import os

DIR = 'data'
FILES = [f'{DIR}/{_}' for _ in os.listdir(DIR) if '.png' in _]

IMAGE_THRESHOLD = ''
IMG_SIZE = ''
IMG_SHRINK = ''
OPTION_1 = {
    'IMAGE_THRESHOLD': 170,
    'IMG_SIZE': numpy.array([145,70]),
    'IMG_SHRINK': 0.4
}
for k, v in OPTION_1.items():
    globals()[k] = v

if len(sys.argv) == 2:
    IMG_SHRINK = float(sys.argv[1])

# Create folder
if os.path.exists(f'{DIR}/{IMG_SHRINK}'):
    os.rmdir(f'{DIR}/{IMG_SHRINK}')
os.mkdir(f'{DIR}/{IMG_SHRINK}')

for file in FILES:
    img = Image.open(file).convert('L').resize(map(int,IMG_SIZE*IMG_SHRINK), Image.Resampling.LANCZOS)
    data = numpy.array(img)
    number = re.search(r'data/(.*).png', file).group(1)
    savePath = f'{DIR}/{IMG_SHRINK}/{number}.txt'

    if os.path.exists(savePath):
        os.remove(savePath)

    with open(savePath, 'a') as file:
        for row in data:
            file.write(' '.join(['O' if i > IMAGE_THRESHOLD else '-' for i in row])+'\n')