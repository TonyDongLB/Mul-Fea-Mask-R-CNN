import pycurl
import json
from io import *
import glob

def getJson():
    img1 = glob.glob('/home/dl/Documents/Hand/1/*jpg')
    img0 = glob.glob('/home/dl/Documents/Hand/0/*jpg')
    c = pycurl.Curl()
    for path in img0:
        file_name = path.split('/')[-1]
        url = 'http://localhost:3000/api/json/%2F0%2F' + file_name
        b = BytesIO()
        c.setopt(pycurl.WRITEFUNCTION, b.write)
        c.setopt(c.URL, url)
        c.perform()
        result = str(b.getvalue().decode("utf-8"))
        if len(result) < 200:
            continue
        else:
            json_file = open('/home/dl/Documents/Hand/jsons/' + file_name + '.json', 'w')
            json_file.write(result)
            json_file.close()
    for path in img1:
        file_name = path.split('/')[-1]
        url = 'http://localhost:3000/api/json/%2F1%2F' + file_name
        b = BytesIO()
        c.setopt(pycurl.WRITEFUNCTION, b.write)
        c.setopt(c.URL, url)
        c.perform()
        result = str(b.getvalue().decode("utf-8"))
        if len(result) < 200:
            continue
        else:
            json_file = open('/home/dl/Documents/Hand/jsons/' + file_name + '.json', 'w')
            json_file.write(result)
            json_file.close()

    try:
        c = pycurl.Curl()
        b = BytesIO()
        c.setopt(pycurl.WRITEFUNCTION, b.write)
        c.setopt(c.URL, 'https://baidu.com')
        c.setopt(pycurl.SSL_VERIFYPEER, 1)
        c.setopt(pycurl.SSL_VERIFYHOST, 2)
        # <TIPS>windows 要指定证书的路径不然会出现(77, "SSL: can't load CA certificate file E:\\curl\\ca-bundle.crt")
        # 证书路径就在curl下载的压缩包里面。mac/linux下面可以注释掉。
        c.setopt(pycurl.CAINFO, "E:\curl\ca-bundle.crt")
        # </TIPS>
        c.perform()
        result = b.getvalue().decode("utf-8")
        print(result)
    except BaseException as e:
        print(e)
    finally:
        b.close()
        c.close()

def testHandset():
    import hand
    import torch
    hand_set = hand.Hand('train', hand.HandConfig())
    train_generator = torch.utils.data.DataLoader(hand_set, batch_size=1, shuffle=True, num_workers=1)
    for result in train_generator:
        a = 1
        continue

if __name__ == '__main__':
    while True:
        testHandset()

