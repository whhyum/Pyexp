import os
import numpy as np
import cv2 as cv
import glob
from tqdm import tqdm

def loadData(localPath, label):
    # 获取本地图片数据
    path = glob.glob(os.path.join(localPath, '*.jpg'))
    x = np.empty([len(path), 224, 224, 3])
    y = np.empty([0])  # 定义x、y变量
    # tqdm 查看进度
    for i in tqdm(range(len(path))):
        imgPath = path[i]  # 遍历图片进行数据赋值
        img = cv.imread(imgPath)

        # 转换通道 => BGR => RGB
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # 转换图形分辨率 224*224
        img = cv.resize(img, (224, 224))

        
    return path

# test = loadData('../data/test', None)
# print(len(test))