import tensorflow as tf
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from keras.preprocessing.image import ImageDataGenerator
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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
    image = tf.image.resize(image, (64, 64))
    image = tf.cast(image, tf.float32)

    feature_dict['image'] = image
    return feature_dict['image'], feature_dict['image/class']


dataset = dataset.map(parese_example)
dataset = dataset.repeat().shuffle(5000).batch(1).prefetch(1)
#
# for img, label in dataset:  # 只取前1条
#     print(label.numpy())
#     print(img.numpy()[0].shape)
#     plt.imshow(np.array(img.numpy()[0], np.uint8))
#     plt.show()

def build_model():
    # based on VGG-16
    preModel_path = "../source/resnet50_weights_tf_dim_ordering_tf_kernels_notop (1).h5"
    ResNet18 = tf.keras.applications.ResNet50(weights=preModel_path, include_top=False)
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
model.load_weights('./weights.h5')

# dataset = ImageDataGenerator(rescale=1. / 255)

batch_size = 8
test_loss, test_acc = model.evaluate_generator(dataset, steps=dataset.img.samples / batch_size)
print('test acc: %.3f%%' % test_acc)