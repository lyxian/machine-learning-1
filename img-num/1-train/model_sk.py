########################################
#  ML model to read number from image  #
########################################

from sklearn.ensemble import RandomForestClassifier

# Get training data
import os
DATA_DIR = 'train'

files = []

for folder in os.listdir(DATA_DIR):
    if folder == '_':
        continue
    _ = [files.append(f'{DATA_DIR}/{folder}/{file}') for file in os.listdir(f'{DATA_DIR}/{folder}')]
print(f'\nTRAIN DATA HAS *{len(files)}* NUMBER OF IMAGES\n')

from PIL import Image

# Get WIDTH, HEIGHT
d = {}
for file in files:
    img = Image.open(f'{file}')
    width, height = img.size
    d[width * height] = f'{width} {height}'
WIDTH, HEIGHT = map(int, d[max(d)].split())

# Save Training Data
import numpy
import re
TRAIN_DIR = 'train'
SAVE_TRAIN_IMG = False

dataArray = []
labelArray = []

for idx, file in enumerate(files, start=1):
    NUMBER = int(re.search(r'.*-(\d)\.*', file).group(1))
    img = Image.open(f'{file}').convert('L').resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    
    data = numpy.array(img) / 255
    dataArray += [data]
    labelArray += [NUMBER]

    if SAVE_TRAIN_IMG:
        img.save(f'{TRAIN_DIR}/{idx}-{NUMBER}.png')
# dataArray = numpy.array(dataArray).reshape(-1, WIDTH, HEIGHT, 1)
dataArray = numpy.array(dataArray).reshape(-1, WIDTH*HEIGHT)

# Hot Code Labelling
DIGITS = [0,1,2,3,4,5,6,7,8,9]
labelArray_1 = []
for label in labelArray:
    _ = numpy.zeros(len(DIGITS))
    _[label] += 1
    labelArray_1 += [_]
labelArray_2 = numpy.array(list(labelArray_1))

# Create ML Model
model = RandomForestClassifier(n_estimators=100)

import time
t_1 = time.time()
history = model.fit( dataArray , labelArray_2 )
print(f'Time Taken: {round(time.time()-t_1,2)} s...')

# Check TRAIN_DATA
if 0:
    train_res = model.predict( dataArray )
    # print(train_res)

    for label, prediction in zip(labelArray, train_res):
        PREDICTED = list(prediction).index(max(prediction))
        print(f'Label {label} = {PREDICTED}')

# Check TEST_DATA
TEST_DIR = 'validation'
if 1:
    files = []
    for folder in os.listdir(TEST_DIR):
        if folder == '_':
            continue
        _ = [files.append(f'{TEST_DIR}/{folder}/{file}') for file in sorted(os.listdir(f'{TEST_DIR}/{folder}'))]
    print(len(files), files)

    dataArray = []
    labelArray = []

    for idx, file in enumerate(files, start=1):
        NUMBER = int(re.search(r'.*-(\d)\.*', file).group(1))
        img = Image.open(f'{file}').convert('L').resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        
        data = numpy.array(img)
        dataArray += [data]
        labelArray += [NUMBER]

    dataArray = numpy.array(dataArray).reshape(-1, WIDTH*HEIGHT)

    # Hot Code Labelling
    DIGITS = [0,1,2,3,4,5,6,7,8,9]
    labelArray_1 = []
    for label in labelArray:
        _ = numpy.zeros(len(DIGITS))
        _[label] += 1
        labelArray_1 += [_]
    labelArray_2 = numpy.array(list(labelArray_1))

    train_res = model.predict( dataArray )
    # print(train_res)

    correct = 0
    for label, prediction in zip(labelArray, train_res):
        PREDICTED = list(prediction).index(max(prediction))
        print(f'Label {label} = {PREDICTED}')
        if label == PREDICTED:
            correct += 1
    print(f'Accuracay: {correct/len(train_res)*100:.2f}%')