from bs4 import BeautifulSoup
from PIL import Image
import subprocess
import requests
import numpy
import time
import sys
import os

FILES = ['train.html', 'train-1.html', 'train-2.html']
DIR = 'data/'

IMAGE = 'train.png'
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

for file in FILES:
    if os.path.exists(file):
        os.remove(file)
        print(f'{file} removed..')

url = 'https://trainarrivalweb.smrt.com.sg/default.aspx'


headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
    'X-MicrosoftAjax': 'Delta=true'    # include to get non-html format
}

if len(sys.argv) == 2:
    n = int(sys.argv[1])
else:
    n = 1

for _ in range(n):

    headers['Cookie'] = getCookieStr(url)
    _, imageUrl, payload = getPayload()
    code = input('Verify train.png , CODE=')
    payload['txtCodeNumber'] = code

    img = Image.open(IMAGE).convert('L').resize(map(int,IMG_SIZE*IMG_SHRINK), Image.Resampling.LANCZOS)
    data = numpy.array(img)

    if os.path.exists('number.txt'):
        os.remove('number.txt')

    with open('number.txt', 'a') as file:
        for row in data:
            file.write(' '.join(['O' if i > IMAGE_THRESHOLD else '-' for i in row])+'\n')

    # Move train.png, number.txt
    subprocess.run(['mv', 'number.txt', f'data/{code}.txt'])
    subprocess.run(['cp', 'train.png', f'data/{code}.png'])
    
    time.sleep(10)