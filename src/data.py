import os
import numpy as np
import cv2 as cv
import glob
from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVIES'] = "0, 1"

def loadData(localPath, label):
    # 获取本地图片数据
    path = glob.glob(os.path.join(localPath, '*.jpg'))
    x = np.empty([len(path), 128, 128, 3])
    y = np.empty([0])  # 定义x、y变量
    # tqdm 查看进度
    for i in tqdm(range(len(path))):
        imgPath = path[i]  # 遍历图片进行数据赋值
        img = cv.imread(imgPath)

        # 转换通道 => BGR => RGB
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # 转换图形分辨率 224*224 64
        img = cv.resize(img, (128, 128))

        x[i, :] = img

    # 填充 y 数据
    y = np.linspace(label, label, x.shape[0])
    return x, y

# test = loadData('../data/test', None)
# print(len(test))

# 定义cat标签为 0，dog标签为 1
trainCatX, trainCatY = loadData('../data/dataset/train/cat', 0)
trainDogX, trainDogY = loadData('../data/dataset/train/dog', 1)
valCatX, valCatY = loadData('../data/dataset/validation/cat', 0)
valDogX, valDogY = loadData('../data/dataset/validation/dog', 1)

# 合并
trainData = np.concatenate((trainCatX, trainDogX), axis=0)
trainLabel = np.concatenate((trainCatY, trainDogY), axis=0)
valData = np.concatenate((valCatX, valDogX), axis=0)
valLabel = np.concatenate((valCatY, valDogY), axis=0)

# 测试
trainImage = trainData[233]
testImage = valData[1888]

trainImage = cv.cvtColor(trainImage.astype(np.uint8), cv.COLOR_RGBA2BGR)
testImage = cv.cvtColor(testImage.astype(np.uint8), cv.COLOR_RGBA2BGR)
cv.imwrite('test1.jpg', trainImage)
cv.imwrite('test2.jpg', testImage)

print(trainLabel[233])
print(valLabel[1888])

np.save('./dataset/trainData.npy', trainData)
np.save('./dataset/valData.npy', valData)
np.save('./dataset/trainLabel.npy', trainLabel)
np.save('./dataset/valLabel.npy', valLabel)