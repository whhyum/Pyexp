#配置图片生成器
from keras.preprocessing.image import ImageDataGenerator

#设置文件目录
#训练集 data\\train\\cat
trainDir ='../dara/train/'
trainDogDir = '../dara/train/dog/'
trainCatDir = '../dara/train/cat/'
# #验证集
# valDir = './check/'
# valDogDir = './check/check_dog/'
# valCatDir = './check/check_cat/'
# #测试集
# testDir = './test/'
# testDogDir = './test/test_dog/'
# testCatDir = './test/test_cat/'


#将图片像素缩小为[0,1]之间的浮点数
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,#图像随机旋转的最大角度
    width_shift_range=0.2,#图片在水平位置上偏移的最大百分比值
    height_shift_range=0.2,#数值位置上
    shear_range=0.2,#随机错位切换的角度
    zoom_range=0.2,#图片随机缩放的范围
    horizontal_flip=True)#随机将一半的图片进行水平翻转

#验证集的数据不能增强
test_datagen = ImageDataGenerator(rescale=1./255)

#创建图片生成器
train_generator = train_datagen.flow_from_directory(
 trainDir,#图片地址
 target_size=(150, 150),#将图片调整为(150,150)大小
 batch_size=32,#设置批量数据的大小为32
 class_mode='binary')#设置返回标签的类型
# val_generator = test_datagen.flow_from_directory(
#  valDir,
#  target_size=(150, 150),
#  batch_size=32,
#  class_mode='binary')