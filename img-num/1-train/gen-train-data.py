from bs4 import BeautifulSoup
from PIL import Image
import subprocess
import requests
import numpy
import sys
import os

if len(sys.argv) == 2:
    MODULE = sys.argv[1].upper()
elif len(sys.argv) == 3 and sys.argv[1] == 'NEW':
    MODULE = sys.argv[1].upper()
else:
    print('Please input MODULE as first argument (NEW, PROCESS, LOCALIZE)')
    sys.exit()

FILES = ['train.html', 'train-1.html', 'train-2.html']
DIR = 'data'
SUB_DIR = 'img'

def getCookieStr(url):
    session = requests.Session()
    response = session.get(url)
    with open('train.html', 'w') as file:
        file.write(response.text)
    if response.ok:
        cookies = session.cookies.get_dict()
        if len(cookies) == 1:
            _, imageUrl, _ = getPayload()
            response = session.get(f'{url.split("/default")[0]}/{imageUrl}')
            with open('train.png', 'wb') as file:
                file.write(response.content)
            return '='.join([j for i in cookies.items() for j in i])
        else:
            return ''
    else:
        return ''

def getPayload():
    with open('train.html', 'r') as file:
        content = file.read()
    soup = BeautifulSoup(content, 'html.parser')
    imageUrl = soup.find('img', {'id': 'imgCaptcha'}).attrs['src']
    d = {
        'ScriptManager1': 'UP1|ibtnSubmit',
        '__EVENTTARGET': 'ibtnSubmit'
    }
    for ___ in soup.find_all('input'):
        if ___.attrs['name'] != '__EVENTTARGET':
            if 'value' in ___.attrs.keys():
                d[___.attrs['name']] = ___.attrs['value']
            else:
                d[___.attrs['name']] = ''
    return soup, imageUrl, d

if MODULE == 'NEW':
    # Check Directory Structure
    for file in FILES:
        if os.path.exists(file):
            os.remove(file)
            print(f'{file} removed..')
    if not os.path.exists(f'{DIR}/{SUB_DIR}'):
        os.mkdir(f'{DIR}/{SUB_DIR}')
        print(f'{DIR}/{SUB_DIR} created..')


    url = 'https://trainarrivalweb.smrt.com.sg/default.aspx'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
        'X-MicrosoftAjax': 'Delta=true'    # include to get non-html format
    }

    if len(sys.argv) == 3:
        n = int(sys.argv[2])
    else:
        n = 1

    counter = 0
    while counter != n:
        headers['Cookie'] = getCookieStr(url)
        _, imageUrl, payload = getPayload()
        code = input(f'Verify train.png ({counter+1}/{n}), CODE=')
        payload['txtCodeNumber'] = code
        if os.path.exists(f'{DIR}/{SUB_DIR}/{code}.png'):
            print(f'Error, \'{code}\' exists, retrying..')
        else:
            subprocess.run(['cp', 'train.png', f'{DIR}/{SUB_DIR}/{code}.png'])
            counter += 1


# =====


from PIL import Image
import subprocess
import numpy
import sys
import re
import os

DIR = 'data'
SUB_DIR = 'img'
FILES = [f'{DIR}/{SUB_DIR}/{_}' for _ in os.listdir(f'{DIR}/{SUB_DIR}') if '.png' in _]

SAVE_TYPE = 'png'
IMAGE_THRESHOLD = 210
IMG_SIZE = numpy.array([145,70])

if MODULE == 'PROCESS':
    # Create folder
    if os.path.exists(f'{DIR}/{SAVE_TYPE}'):
        subprocess.run(['rm', '-r', f'{DIR}/{SAVE_TYPE}'])
    os.mkdir(f'{DIR}/{SAVE_TYPE}')

    for file in FILES:
        img = Image.open(file).convert('L').resize(map(int,IMG_SIZE), Image.Resampling.LANCZOS)
        data = numpy.array(img)
        number = re.search(rf'{DIR}/{SUB_DIR}/(.*).png', file).group(1)
        savePath = f'{DIR}/{SAVE_TYPE}/{number}.{SAVE_TYPE}'

        if os.path.exists(savePath):
            os.remove(savePath)

        if SAVE_TYPE == 'png':
            newData = numpy.array([[0 if col > IMAGE_THRESHOLD else 255 for col in row] for row in data]).astype(numpy.uint8)
            Image.fromarray(newData).save(savePath)


# =====


from random import sample
from PIL import Image
import numpy as np
import cv2
import os

n = 130

EXPAND = 4
KERNEL_SIZE = (4, 4)
REQUIRED_DIR = ['train', 'validation'] # + ['flagged']
TRAIN_SOURCE_DIR = f'data/png'
TRAIN_DIR = 'train'
TEST_DIR = 'validation'
TEST_SIZE = 10

if MODULE == 'LOCALIZE':
    FILES = sample(os.listdir(TRAIN_SOURCE_DIR), len(os.listdir(TRAIN_SOURCE_DIR)))[:n]
    
    # Create required folders
    for TMP_DIR in REQUIRED_DIR:
        if os.path.exists(TMP_DIR):
            subprocess.run(['rm', '-r', TMP_DIR])
            print(f'{TMP_DIR} removed..')
        os.mkdir(TMP_DIR)
        print(f'{TMP_DIR} created..')

    for FILENAME in FILES:
        NUMBER = FILENAME.split('.')[0]
        SAVE_PATH = f'train/{NUMBER}'
        SAVE_IMAGE = f'{SAVE_PATH}/0.png'

        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)
            print(f'{SAVE_PATH} created..')
        else:
            print(f'Folder {SAVE_PATH} exists')
            import sys
            sys.exit()

        # Process Image
        if 1:
            img = cv2.imread(f'{TRAIN_SOURCE_DIR}/{FILENAME}')
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
                if x < 0 or y < 0:   # exclude NEGATIVES
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

            btmLeft = list(zip(x1, [y1]*3))
            topRight = list(zip(x2, [y2]*3))

            if len(btmLeft) == 3 and len(topRight) == 3:
                for idx, (x1y1, x2y2) in enumerate(zip(btmLeft, topRight), start=1):
                    x1, y1 = x1y1; x2, y2 = x2y2
                    digit = NUMBER[idx-1]
                    cv2.imwrite(f'{SAVE_PATH}/{idx}-{digit}.png', img[y1:y2,x1:x2])
                    print(f'Saved to {SAVE_PATH}/{idx}-{digit}.png ...')
                os.remove(SAVE_IMAGE)
            else:
                print(f'{NUMBER} not localized properly..')
                subprocess.run(['cp', SAVE_IMAGE, f'flagged/{NUMBER}.png'])
                os.remove(SAVE_IMAGE)
                os.rmdir(SAVE_PATH)

        print(f'{SAVE_PATH} removed..')

    # Move validation data
    TEST_FOLDERS = sample(os.listdir(TRAIN_DIR), len(os.listdir(TRAIN_DIR)))[:TEST_SIZE]
    for FOLDER in TEST_FOLDERS:
        subprocess.run(['mv', f'{TRAIN_DIR}/{FOLDER}', f'{TEST_DIR}/'])
        print(f'{FOLDER} moved to validation/ ..')