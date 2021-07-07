import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report,confusion_matrix
# from sklearn.metrics import confusion_matrix

# from keras.preprocessing.image import ImageDataGenerator
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# tf.enable_eager_execution(
#     config=None,
#     device_policy=None,
#     execution_mode=None
# )

# 测试模型最终效果

def build_model():
    """
    使用weight重新加载模型
    :return:
    """
    path = '../source/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    ResNet18 = tf.keras.applications.ResNet50(weights=path, include_top=False)
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    fc = tf.keras.layers.Dense(2, activation="softmax")  # 修改乘自己的类别数
    model = tf.keras.Sequential([ResNet18, global_average_layer, fc])
    # configure the model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary()
    return model

model = build_model()
model.load_weights('./model/weights6.19.h5')

test_record_path = "./dataset/test.record"

dataset = tf.data.TFRecordDataset(test_record_path)

feature_description = {
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/class': tf.io.FixedLenFeature([], tf.int64),
    'image/encoded': tf.io.FixedLenFeature([], tf.string)
}

def parese_example(serialized_example):
    feature_dict = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.decode_jpeg(feature_dict['image/encoded'])  # 解码JPEG图片
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32)

    feature_dict['image'] = image
    return feature_dict['image'], feature_dict['image/class']


dataset = dataset.map(parese_example)
dataset = dataset.repeat().shuffle(5000).batch(1).prefetch(1)
#
y_true = []
y_pred = []
for img, label in dataset.take(1000):
    # print(label.numpy())
    # print(img.numpy()[0].shape)
    y_true.append(label.numpy()[0])
    print('true',y_true)
    x = np.expand_dims(img, 0)
    y = model.predict(img)
    y_pred.append(np.argmax(y, axis=1)[0])
    # y_pred = y_pred
    print('pred', y_pred)
    if(label.numpy()[0] != np.argmax(y, axis=1)[0]):
        predict_cat = y[:, 0]
        predict_dog = y[:, 1]
        print('猫的概率：',predict_cat)
        print('狗的概率', predict_dog)
        # plt.figure(1)
        # plt.imshow(np.array(img.numpy()[0], np.uint8))
        # plt.show()


# dataset = ImageDataGenerator(rescale=1. / 255)
cm=confusion_matrix(y_true,y_pred)
print('混淆矩阵: ')
print( cm)
print(classification_report(y_true,y_pred,target_names=['cat','dog']))

def plot_confusion_matrix(cm, labels_name, title):
    np.set_printoptions(precision=2)
    # print(cm)
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # show confusion matrix
    plt.savefig('./dataset/tt.png', format='png')
    plt.show()

# print('type=',type(cm))
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
# print(cm)
labels = [0, 1]  #类别集合
plot_confusion_matrix(cm,labels,'confusion_matrix')

# batch_size = 8
# test_loss, test_acc = model.evaluate_generator(dataset, steps = 10)
# print('test acc: %.3f%%' % test_acc)